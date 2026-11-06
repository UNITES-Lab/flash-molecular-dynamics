"""Captures golden MACE energies and forces from the FlashMACE fork.

Every MACE number in the FlashMD paper came from
`scripts/benchmark_mace_protein_md.py` in the FlashMACE fork (mace-torch
0.3.15 plus the `dev/flash-mace` commits). Before that fork is replaced by an
adapter over stock mace-torch, this script evaluates it once, without
dynamics, on fixed perturbed inputs and stores the results as goldens.

The arms are built with the benchmark's own `parse_args`, `load_model`,
`build_batch_template`, `maybe_compile_*` and `run_forward`, so they are the
paper's arms rather than copies of them:

* `a_baseline_cueq_f64`: the paper's reference, `--backend cueq_f64`.
* `b_stock_cueq_fp32`: `--backend cueq_fp32` with the fork's global Triton
  geometry hook disabled, that is stock cuEquivariance MACE in float32.
* `c_flashmace`: the paper's FlashMACE arm, `--backend cueq_triton_fp32
  --allow-tf32 --matmul-precision high --compile-core-modules
  --fixed-onehot-fctp --energy-force-only`.
* `d_flashmace_no_triton_geometry`: arm c with
  `MACE_DISABLE_TRITON_GEOMETRY=1`.
* `e_flashmace_no_tf32`: arm c without TF32, which separates TF32 round-off
  from the effect of the other switches.
* `f_e3nn_f64`: `--backend e3nn_f64`, stock e3nn MACE in float64, which
  checks the cuEquivariance conversion of arm a.

Each (input, arm) pair runs in a fresh subprocess, like one benchmark call in
`--mode single`, so process-wide state (default dtype, TF32 flags, Dynamo
caches, the geometry environment variable) cannot leak between arms. The
driver then writes `summary.json` with versions, flags and pairwise
differences computed in float64.

Usage, on a GPU node with the `flash-mace` conda environment:

    python mace_goldens.py --flash-mace "$FLASH_MACE" --out-dir "$GOLDEN_DIR"
"""

import argparse
from collections.abc import Mapping, Sequence
import dataclasses
import hashlib
import importlib
import importlib.metadata
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch

_DEFAULT_FLASH_MACE = pathlib.Path(
    "/playpen-shared/pingzhi/efficient-schnet/flash-mace"
)
_REPLICAS = 4
_NOISE_ANGSTROM = 0.01
_NOISE_SEED = 20261105
_REPEATS = 3
_GEOMETRY_ENV = "MACE_DISABLE_TRITON_GEOMETRY"


@dataclasses.dataclass(frozen=True)
class GoldenInput:
    """A structure that every arm evaluates.

    Attributes:
        name: Short name used in output file names.
        pdb: Path relative to the parent of the FlashMACE checkout, exactly
            as the paper's run read it.
        num_atoms: Atoms per replica recorded by the paper's run.
        num_edges: Edges per replica recorded by the paper's run.
        elements: Element counts recorded by the paper's run.
        paper_run: Result file of the paper's run on this input.
    """

    name: str
    pdb: str
    num_atoms: int
    num_edges: int
    elements: Mapping[str, int]
    paper_run: str


@dataclasses.dataclass(frozen=True)
class Arm:
    """One configuration of the benchmark script.

    Attributes:
        name: Short name used in output file names.
        description: What the arm stands for.
        flags: Flags passed to the benchmark's `parse_args`.
        env: Environment variables set in the arm's subprocess.
    """

    name: str
    description: str
    flags: tuple[str, ...]
    env: Mapping[str, str] = dataclasses.field(default_factory=dict)


_INPUTS = (
    GoldenInput(
        name="2jof_5beads",
        pdb="flash-schnet/data/simulating_a_trained_cg_model/2jof_5beads.pdb",
        num_atoms=97,
        num_edges=2384,
        elements={"C": 57, "N": 20, "O": 20},
        paper_run=(
            "flash-mace/data/profile_results/"
            "2jof_bs64_paired_replay_static_fixed_energyforce_tf32_"
            "compilecore_10k.json"
        ),
    ),
    GoldenInput(
        name="chignolin_native",
        pdb=(
            "submission/data/protein_size_scaling/inputs/mace/"
            "chignolin_native.pdb"
        ),
        num_atoms=166,
        num_edges=8380,
        elements={"H": 73, "C": 62, "N": 11, "O": 20},
        paper_run=(
            "flash-mace/data/protein_size_scaling/results_bs64/mace/"
            "{baseline,flash}/chignolin.json"
        ),
    ),
)

_FLASH_SWITCHES = (
    "--compile-core-modules",
    "--fixed-onehot-fctp",
    "--energy-force-only",
)
_TF32 = ("--allow-tf32", "--matmul-precision", "high")

_ARMS = (
    Arm(
        name="a_baseline_cueq_f64",
        description="Paper reference arm: cuEquivariance, float64.",
        flags=("--backend", "cueq_f64"),
    ),
    Arm(
        name="b_stock_cueq_fp32",
        description=(
            "Precision-matched stock arm: cuEquivariance, float32, no "
            "FlashMACE switches, Triton geometry hook off, TF32 off."
        ),
        flags=("--backend", "cueq_fp32"),
        env={_GEOMETRY_ENV: "1"},
    ),
    Arm(
        name="c_flashmace",
        description="Paper FlashMACE arm, all switches and TF32 as used.",
        flags=("--backend", "cueq_triton_fp32", *_TF32, *_FLASH_SWITCHES),
    ),
    Arm(
        name="d_flashmace_no_triton_geometry",
        description="Arm c with the global Triton geometry hook disabled.",
        flags=("--backend", "cueq_triton_fp32", *_TF32, *_FLASH_SWITCHES),
        env={_GEOMETRY_ENV: "1"},
    ),
    Arm(
        name="e_flashmace_no_tf32",
        description="Arm c with TF32 off (matmul precision highest).",
        flags=("--backend", "cueq_triton_fp32", *_FLASH_SWITCHES),
    ),
    Arm(
        name="f_e3nn_f64",
        description="Stock e3nn MACE in float64, as the benchmark loads it.",
        flags=("--backend", "e3nn_f64"),
    ),
)

_PAIRS = (
    ("a_baseline_cueq_f64", "b_stock_cueq_fp32"),
    ("a_baseline_cueq_f64", "c_flashmace"),
    ("b_stock_cueq_fp32", "c_flashmace"),
    ("c_flashmace", "d_flashmace_no_triton_geometry"),
    ("b_stock_cueq_fp32", "e_flashmace_no_tf32"),
    ("e_flashmace_no_tf32", "c_flashmace"),
    ("f_e3nn_f64", "a_baseline_cueq_f64"),
)
_CUDA_LIBRARIES = ("libcue_ops", "libcublas", "libnvrtc", "libcudart")

_PROVENANCE = {
    "checkpoint": "MACE-MP-0 medium (20231203mace128L1_epoch199model)",
    "headline_10k": {
        "result": _INPUTS[0].paper_run,
        "report": "flash-mace/data/profile_results/"
        "MACE_PROTEIN_MD_10K_REPORT.md",
        "flags": "--mode paired --paired-schedule replay "
        "--baseline-backend cueq_f64 --candidate-backend cueq_triton_fp32 "
        "--candidate-fixed-onehot-fctp --candidate-energy-force-only "
        "--compile-candidate-core-modules --allow-tf32 --batch-size 64 "
        "--steps 10000 --measure-window 1000",
    },
    "protein_size_scaling": {
        "runner": "submission/scripts/run_protein_size_scaling.py:run_mace",
        "baseline_flags": "--mode single --backend cueq_f64 --steps 12 "
        "--measure-window 8",
        "flash_flags": "--mode single --backend cueq_triton_fp32 "
        "--allow-tf32 --matmul-precision high --compile-core-modules "
        "--fixed-onehot-fctp --energy-force-only --steps 12 "
        "--measure-window 8",
    },
    "geometry_hook": "mace/modules/utils.py:get_edge_vectors_and_lengths "
    "routes every float32 CUDA call to Triton unless "
    f"{_GEOMETRY_ENV}=1; no benchmark flag controls it.",
}

_METRICS = {
    "units": "energy in eV, forces in eV/A",
    "reference": "the first arm of each pair",
    "energy.max_abs": "max_r |E_other - E_ref|",
    "energy.max_rel": "max_r |E_other - E_ref| / |E_ref|",
    "energy.max_abs_per_atom": "energy.max_abs / atoms per replica",
    "forces.max_abs": "max over all components |F_other - F_ref|",
    "forces.max_rel": "forces.max_abs / max |F_ref|",
    "forces.relative_l2": "||F_other - F_ref||_2 / ||F_ref||_2 "
    "(benchmark force_error)",
    "forces.relative_max_norm": "max_i |dF_i| / max_i |F_ref,i| over atom "
    "vectors (benchmark force_error, the paper's accuracy gate)",
    "repeat_spread": f"max abs change across {_REPEATS} identical calls in "
    "one process; the last call is stored",
    "positions": "stored float64 inputs, [R, N, 3]; float32 arms evaluate "
    "them rounded to float32",
    "edge_index": "one replica's fixed graph, built by build_batch_template "
    "from the unperturbed structure and replicated with atom offsets",
}


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parses command-line arguments.

    Args:
        argv: Arguments without the program name, or None for `sys.argv`.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Capture golden MACE outputs from the FlashMACE fork."
    )
    parser.add_argument(
        "--flash-mace", type=pathlib.Path, default=_DEFAULT_FLASH_MACE
    )
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Skip the arms and rebuild summary.json from existing files.",
    )
    parser.add_argument(
        "--worker",
        nargs=2,
        metavar=("INPUT", "ARM"),
        help="Internal: evaluate one arm on one input in this process.",
    )
    return parser.parse_args(argv)


def _by_name(items: Sequence[Any], name: str) -> Any:
    """Returns the item whose `name` attribute equals `name`.

    Args:
        items: Objects with a `name` attribute.
        name: Name to look up.

    Returns:
        The matching item.

    Raises:
        KeyError: If no item has that name.
    """
    for item in items:
        if item.name == name:
            return item
    raise KeyError(name)


def _output_path(
    out_dir: pathlib.Path, input_name: str, arm: str
) -> pathlib.Path:
    """Returns the golden file path for one input and arm."""
    return out_dir / f"{input_name}_{arm}.pt"


def _sha256(path: pathlib.Path) -> str:
    """Returns the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _import_benchmark(flash_mace: pathlib.Path) -> Any:
    """Imports the FlashMACE benchmark script as a module.

    The benchmark puts the checkout at the front of `sys.path` on import, so
    `mace` must then resolve to that checkout.

    Args:
        flash_mace: Root of the FlashMACE checkout.

    Returns:
        The imported `benchmark_mace_protein_md` module.

    Raises:
        RuntimeError: If `mace` is imported from somewhere else.
    """
    scripts = str(flash_mace / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    bench = importlib.import_module("benchmark_mace_protein_md")
    import mace

    mace_file = pathlib.Path(mace.__file__).resolve()
    if flash_mace.resolve() not in mace_file.parents:
        raise RuntimeError(f"mace imported from {mace_file}, not {flash_mace}")
    return bench


def _benchmark_args(
    bench: Any, pdb: pathlib.Path, arm: Arm
) -> tuple[argparse.Namespace, list[str]]:
    """Parses an arm's flags with the benchmark's own parser.

    Args:
        bench: The imported benchmark module.
        pdb: Input structure.
        arm: The arm to configure.

    Returns:
        The benchmark namespace and the argument list that produced it.
    """
    argv = [
        "--pdb",
        str(pdb),
        "--model-size",
        "medium",
        "--batch-size",
        str(_REPLICAS),
        "--mode",
        "single",
        "--output",
        os.devnull,
        *arm.flags,
    ]
    saved = sys.argv
    sys.argv = ["benchmark_mace_protein_md.py", *argv]
    try:
        return bench.parse_args(), argv
    finally:
        sys.argv = saved


def _apply_process_settings(args: argparse.Namespace) -> str:
    """Applies the process-wide settings of the benchmark's `main`.

    Args:
        args: The benchmark namespace.

    Returns:
        The float32 matmul precision that was set.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.allow_tf32 = args.allow_tf32
    precision = args.matmul_precision or (
        "high" if args.allow_tf32 else "highest"
    )
    torch.set_float32_matmul_precision(precision)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    return precision


def _noisy_positions(base: np.ndarray) -> torch.Tensor:
    """Returns `_REPLICAS` copies of `base` with fixed Gaussian noise.

    Args:
        base: Positions of one replica in Angstrom, shape [N, 3].

    Returns:
        Float64 CPU positions of shape [R, N, 3], identical for every arm.
    """
    positions = torch.as_tensor(base, dtype=torch.float64)
    generator = torch.Generator(device="cpu").manual_seed(_NOISE_SEED)
    noise = torch.randn(
        (_REPLICAS, *positions.shape), generator=generator, dtype=torch.float64
    )
    return positions.unsqueeze(0) + _NOISE_ANGSTROM * noise


def _introspect(
    model: torch.nn.Module, template: Mapping[str, Any]
) -> dict[str, Any]:
    """Records which FlashMACE switches are live in a built model.

    Args:
        model: The model after `load_model` and before compilation.
        template: The batch template built for the model.

    Returns:
        JSON-serializable switch states.
    """
    from mace.modules import fixed_onehot_fctp
    from mace.modules import triton_radial
    from mace.modules import utils

    products = list(getattr(model, "products", []))
    return {
        "model_class": type(model).__name__,
        "parameter_dtype": str(next(model.parameters()).dtype),
        "default_dtype": str(torch.get_default_dtype()),
        "triton_radial": isinstance(
            model.radial_embedding, triton_radial.TritonRadialEmbeddingBlock
        ),
        "fixed_onehot_fctp_modules": sum(
            isinstance(m, fixed_onehot_fctp.FixedOneHotFCTP)
            for m in model.modules()
        ),
        "energy_force_only": hasattr(
            model, "_energy_force_only_original_forward"
        ),
        "static_product_indices": bool(products)
        and all(hasattr(p, "_fixed_node_attr_indices") for p in products),
        "triton_geometry_active": bool(
            utils.can_use_triton_edge_geometry(
                template["positions"],
                template["edge_index"],
                template["shifts"],
                False,
            )
        ),
    }


def _mapped_libraries() -> dict[str, list[str]]:
    """Returns the CUDA libraries mapped into this process, by prefix.

    `cuequivariance_ops` only prints a warning when `libcue_ops.so` fails to
    load, so the worker checks the mapping instead of trusting the import.

    Returns:
        For each prefix in `_CUDA_LIBRARIES`, the mapped file paths.
    """
    paths = set()
    for line in pathlib.Path("/proc/self/maps").read_text().splitlines():
        fields = line.split()
        if len(fields) >= 6 and fields[5].startswith("/"):
            paths.add(fields[5])
    return {
        prefix: sorted(
            p for p in paths if pathlib.Path(p).name.startswith(prefix)
        )
        for prefix in _CUDA_LIBRARIES
    }


def _max_spread(values: Sequence[torch.Tensor]) -> float:
    """Returns the largest change of any element across repeated calls."""
    last = values[-1].to(torch.float64)
    return max(float((v.to(torch.float64) - last).abs().max()) for v in values)


def _run_worker(
    flash_mace: pathlib.Path,
    out_dir: pathlib.Path,
    input_name: str,
    arm_name: str,
) -> None:
    """Evaluates one arm on one input and saves the golden file.

    Mirrors the `--mode single` branch of the benchmark's `main`, with the
    dynamics loop replaced by `_REPEATS` calls of `run_forward` on fixed
    perturbed positions.

    Args:
        flash_mace: Root of the FlashMACE checkout.
        out_dir: Directory for the golden files.
        input_name: Name of a `GoldenInput`.
        arm_name: Name of an `Arm`.

    Raises:
        RuntimeError: If CUDA is unavailable or the arm's environment is
            not what the arm requires.
    """
    golden_input = _by_name(_INPUTS, input_name)
    arm = _by_name(_ARMS, arm_name)
    if os.environ.get(_GEOMETRY_ENV) != arm.env.get(_GEOMETRY_ENV):
        raise RuntimeError(f"{_GEOMETRY_ENV} does not match arm {arm.name}")
    # Import first: the benchmark preloads CUDA 13 NVRTC before any CUDA use.
    bench = _import_benchmark(flash_mace)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    pdb = flash_mace.parent / golden_input.pdb
    args, argv = _benchmark_args(bench, pdb, arm)
    precision = _apply_process_settings(args)
    device = torch.device(args.device)
    dtype = bench.backend_dtype(args.backend)

    atoms = bench.load_protein(args.pdb)
    model = bench.load_model(
        bench.model_path(args),
        args.backend,
        device,
        cueq_linear_method=args.cueq_linear_method,
        cueq_fctp_method=args.cueq_fctp_method,
        cueq_channelwise_method=args.cueq_channelwise_method,
        cueq_symmetric_method=args.cueq_symmetric_method,
        cueq_conv_fusion_value=bench.cueq_conv_fusion(args.cueq_conv_fusion),
        cueq_spherical_harmonics=args.cueq_spherical_harmonics,
        fixed_onehot_fctp=args.fixed_onehot_fctp,
        radial_mlp_compute_dtype=bench.radial_mlp_dtype(args.radial_mlp_dtype),
        energy_force_only=args.energy_force_only,
    )
    template, _, system = bench.build_batch_template(
        atoms, model, args.batch_size, dtype, device
    )
    switches = _introspect(model, template)
    model = bench.maybe_compile_core_modules(
        model, args.compile_core_modules, args.compile_mode
    )
    model = bench.maybe_compile_radial_mlps(
        model, args.compile_radial_mlp, args.compile_mode
    )
    model = bench.maybe_compile_model(
        model, args.compile_candidate, args.compile_mode
    )
    switches["compiled_core_modules"] = sum(
        type(m).__name__ == "OptimizedModule"
        for m in [*model.interactions, *model.products]
    )

    positions = _noisy_positions(atoms.positions)
    flat = positions.reshape(-1, 3).to(device)
    energies, forces, times_ms = [], [], []
    for _ in range(_REPEATS):
        energy, force, elapsed_ms = bench.run_forward(
            model, template, flat, dtype=dtype
        )
        energies.append(energy.cpu())
        forces.append(force.cpu())
        times_ms.append(elapsed_ms)
    libraries = _mapped_libraries()
    if args.backend.startswith("cueq") and not libraries["libcue_ops"]:
        raise RuntimeError("libcue_ops.so is not loaded; cuEq CUDA kernels off")

    num_atoms = len(atoms)
    num_edges = system["num_edges_per_replica"]
    metadata = {
        "input": golden_input.name,
        "pdb": str(pdb),
        "arm": arm.name,
        "description": arm.description,
        "benchmark_argv": argv,
        "env": {_GEOMETRY_ENV: os.environ.get(_GEOMETRY_ENV)},
        "backend_dtype": str(dtype),
        "default_dtype_at_eval": str(torch.get_default_dtype()),
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": precision,
        "checkpoint": bench.model_path(args),
        "system": system,
        "switches": switches,
        "noise": {
            "sigma_angstrom": _NOISE_ANGSTROM,
            "seed": _NOISE_SEED,
            "replicas": _REPLICAS,
        },
        "repeats": _REPEATS,
        "repeat_spread": {
            "energy_max_abs": _max_spread(energies),
            "forces_max_abs": _max_spread(forces),
        },
        "forward_ms": times_ms,
        "mace_file": sys.modules["mace"].__file__,
        "benchmark_file": bench.__file__,
        "gpu_name": torch.cuda.get_device_name(device),
        "cuda_libraries": libraries,
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    record = {
        "energy": energies[-1],
        "forces": forces[-1].reshape(_REPLICAS, num_atoms, 3),
        "positions": positions,
        "atomic_numbers": torch.as_tensor(atoms.numbers, dtype=torch.int64),
        "edge_index": template["edge_index"][:, :num_edges].cpu(),
        "cutoff_angstrom": system["cutoff_angstrom"],
        "metadata": metadata,
    }
    torch.save(record, _output_path(out_dir, golden_input.name, arm.name))
    print(json.dumps(metadata), flush=True)


def _compare(
    bench: Any, reference: Mapping[str, Any], other: Mapping[str, Any]
) -> dict[str, dict[str, float]]:
    """Returns energy and force differences between two arms in float64.

    Args:
        bench: The imported benchmark module, for its `force_error`.
        reference: Golden record of the reference arm.
        other: Golden record of the compared arm.

    Returns:
        Metrics as described in `_METRICS`.
    """
    e_ref = reference["energy"].to(torch.float64)
    e_diff = (other["energy"].to(torch.float64) - e_ref).abs()
    f_ref = reference["forces"].to(torch.float64).reshape(-1, 3)
    f_other = other["forces"].to(torch.float64).reshape(-1, 3)
    f_diff = (f_other - f_ref).abs()
    gate = bench.force_error(f_ref, f_other)
    num_atoms = reference["forces"].shape[1]
    return {
        "energy": {
            "max_abs": float(e_diff.max()),
            "max_rel": float((e_diff / e_ref.abs()).max()),
            "max_abs_per_atom": float(e_diff.max()) / num_atoms,
        },
        "forces": {
            "max_abs": float(f_diff.max()),
            "max_rel": float(f_diff.max() / f_ref.abs().max()),
            "relative_l2": gate["relative_l2"],
            "relative_max_norm": gate["relative_max_norm"],
        },
    }


def _edge_set_check(record: Mapping[str, Any]) -> dict[str, int]:
    """Compares the fixed benchmark graph with graphs of the noisy inputs.

    The benchmark builds its edges once from the unperturbed structure and
    replicates them, as during its dynamics. A stock adapter that rebuilds
    the neighbor list from the noisy positions sees these differences.

    Args:
        record: Any golden record of the input.

    Returns:
        Edge counts summed over replicas.
    """
    from mace.data.neighborhood import get_neighborhood

    cutoff = record["cutoff_angstrom"]
    fixed = record["edge_index"].numpy()
    fixed_pairs = set(zip(fixed[0].tolist(), fixed[1].tolist(), strict=True))
    added = removed = beyond = 0
    for replica in record["positions"].numpy():
        edges, *_ = get_neighborhood(
            positions=replica, cutoff=cutoff, pbc=np.array([False] * 3)
        )
        pairs = set(zip(edges[0].tolist(), edges[1].tolist(), strict=True))
        added += len(pairs - fixed_pairs)
        removed += len(fixed_pairs - pairs)
        lengths = np.linalg.norm(replica[fixed[1]] - replica[fixed[0]], axis=1)
        beyond += int((lengths >= cutoff).sum())
    return {
        "fixed_edges_per_replica": len(fixed_pairs),
        "noisy_graph_extra_edges": added,
        "noisy_graph_missing_edges": removed,
        "fixed_edges_beyond_cutoff": beyond,
    }


def _summarize_input(
    bench: Any, out_dir: pathlib.Path, golden_input: GoldenInput
) -> dict[str, Any]:
    """Builds the summary block of one input.

    Args:
        bench: The imported benchmark module.
        out_dir: Directory with the golden files.
        golden_input: The input to summarize.

    Returns:
        Per-arm metadata, input checks and pairwise differences.
    """
    records = {}
    for arm in _ARMS:
        path = _output_path(out_dir, golden_input.name, arm.name)
        if path.exists():
            records[arm.name] = torch.load(path, weights_only=True)
    if not records:
        return {"arms": {}, "checks": {"complete": False}}

    first = next(iter(records.values()))
    system = first["metadata"]["system"]
    checks = {
        "complete": len(records) == len(_ARMS),
        "same_inputs_all_arms": all(
            torch.equal(r[key], first[key])
            for r in records.values()
            for key in ("positions", "atomic_numbers", "edge_index")
        ),
        "matches_paper_system": (
            system["num_atoms_per_replica"] == golden_input.num_atoms
            and system["num_edges_per_replica"] == golden_input.num_edges
            and system["elements"] == dict(golden_input.elements)
        ),
    }
    pairs = {
        f"{ref} vs {other}": _compare(bench, records[ref], records[other])
        for ref, other in _PAIRS
        if ref in records and other in records
    }
    pdb = pathlib.Path(first["metadata"]["pdb"])
    return {
        "pdb": str(pdb),
        "pdb_sha256": _sha256(pdb),
        "paper_run": golden_input.paper_run,
        "system": system,
        "checks": checks,
        "graph": _edge_set_check(first),
        "arms": {name: r["metadata"] for name, r in records.items()},
        "pairs": pairs,
    }


def _git(repo: pathlib.Path, *args: str) -> str:
    """Runs a read-only git command and returns its stripped output."""
    command = ["git", "--no-optional-locks", "-C", str(repo), *args]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError) as error:
        return f"unavailable: {error}"
    return result.stdout.strip()


def _versions(flash_mace: pathlib.Path) -> dict[str, Any]:
    """Collects package, repository and GPU versions.

    Args:
        flash_mace: Root of the FlashMACE checkout.

    Returns:
        JSON-serializable version information.
    """
    packages = {}
    for name in (
        "torch",
        "triton",
        "e3nn",
        "cuequivariance",
        "cuequivariance-torch",
        "cuequivariance-ops-torch-cu12",
        "mace-torch",
        "ase",
        "numpy",
        "openequivariance",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,uuid",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as error:
        smi = [f"unavailable: {error}"]
    status = _git(flash_mace, "status", "--porcelain", "--untracked-files=no")
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "packages": packages,
        "mace_commit": _git(flash_mace, "rev-parse", "HEAD"),
        "mace_branch": _git(flash_mace, "rev-parse", "--abbrev-ref", "HEAD"),
        "mace_tracked_changes": status.splitlines(),
        "nvidia_smi": smi,
        "host": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


def _summarize(
    flash_mace: pathlib.Path,
    out_dir: pathlib.Path,
    failures: Mapping[str, int],
) -> dict[str, Any]:
    """Builds `summary.json` from the golden files.

    Args:
        flash_mace: Root of the FlashMACE checkout.
        out_dir: Directory with the golden files.
        failures: Exit codes of failed workers, keyed by `input/arm`.

    Returns:
        The summary.
    """
    bench = _import_benchmark(flash_mace)
    checkpoint = pathlib.Path(bench.MODEL_PATHS["medium"])
    inputs = {
        golden_input.name: _summarize_input(bench, out_dir, golden_input)
        for golden_input in _INPUTS
    }
    complete = not failures and all(
        block["checks"].get("complete")
        and block["checks"].get("same_inputs_all_arms")
        and block["checks"].get("matches_paper_system")
        for block in inputs.values()
    )
    return {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "complete": complete,
        "failures": dict(failures),
        "versions": _versions(flash_mace),
        "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
        "provenance": _PROVENANCE,
        "arms": {
            arm.name: {
                "description": arm.description,
                "benchmark_flags": list(arm.flags),
                "env": dict(arm.env),
            }
            for arm in _ARMS
        },
        "metric_definitions": _METRICS,
        "inputs": inputs,
    }


def _run_driver(args: argparse.Namespace) -> int:
    """Runs every arm in a subprocess and writes `summary.json`.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Process exit code: 0 if every arm ran and every input check passed.
    """
    args.out_dir.mkdir(parents=True, exist_ok=True)
    failures: dict[str, int] = {}
    if not args.summarize_only:
        script = str(pathlib.Path(__file__).resolve())
        for golden_input in _INPUTS:
            for arm in _ARMS:
                env = os.environ.copy()
                env.pop(_GEOMETRY_ENV, None)
                env.update(arm.env)
                command = [
                    sys.executable,
                    script,
                    "--flash-mace",
                    str(args.flash_mace),
                    "--out-dir",
                    str(args.out_dir),
                    "--worker",
                    golden_input.name,
                    arm.name,
                ]
                # A stale file from an earlier attempt must not survive a
                # failed run of the same arm.
                output = _output_path(args.out_dir, golden_input.name, arm.name)
                output.unlink(missing_ok=True)
                start = time.perf_counter()
                result = subprocess.run(command, env=env, check=False)
                seconds = time.perf_counter() - start
                print(
                    f"[goldens] {golden_input.name} {arm.name}: "
                    f"exit {result.returncode} in {seconds:.0f} s",
                    flush=True,
                )
                if result.returncode != 0:
                    failures[f"{golden_input.name}/{arm.name}"] = (
                        result.returncode
                    )
    summary = _summarize(args.flash_mace, args.out_dir, failures)
    path = args.out_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[goldens] wrote {path}; complete={summary['complete']}")
    return 0 if summary["complete"] else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the driver, or one worker when `--worker` is given.

    Args:
        argv: Arguments without the program name, or None for `sys.argv`.

    Returns:
        Process exit code.
    """
    args = _parse_args(argv)
    if args.worker:
        _run_worker(args.flash_mace, args.out_dir, *args.worker)
        return 0
    return _run_driver(args)


if __name__ == "__main__":
    sys.exit(main())
