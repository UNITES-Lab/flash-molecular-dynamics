"""Captures golden BioEmu outputs before FlashBioEmu becomes an adapter.

The FlashBioEmu fork (microsoft/bioemu 53e73d7 plus the FlashBioEmu commits)
produced the paper's BioEmu throughput numbers and the cryptic-pocket samples.
This tool records what each code path returns for one fixed input, so that the
unified package, an adapter over PyPI `bioemu==1.4.1`, can be tested against
it. The input is ubiquitin (76 residues, the paper's cached embeddings), model
bioemu-v1.1, batch 32, seed 20260616 and the DPM sampler with 9 steps (N=10),
as in `run_protein_size_scaling.py`.

Arms. The letters are the ones used in the summary:

  a          official_eval             upstream `bioemu.sample.generate_batch`,
                                       which calls `bioemu.denoiser.dpm_solver`,
                                       with the score model in eval mode.
  b          official_train            the same call in train mode, dropout
                                       active. Upstream `bioemu.sample.main`
                                       never calls `.eval()`, so this is how
                                       upstream samples.
  a_harness  official_harness_eval     `run_once` of the official benchmark
                                       harness, the paper's baseline series.
  c          flash_accelerated         `CudaGraphDpmRunner` of the fork's
                                       benchmark harness with the static context
                                       cache, the paper's flash series
                                       (`--static-context-cache` and
                                       `--cuda-graph-dpm`).
  c_eager    flash_static_cache_eager  the fork's `run_once` with the static
                                       context cache and no CUDA graph.
  c_sample   flash_sample_accelerated  the fork's `generate_batch` with both
                                       flags on, the entry point of the
                                       cryptic-pocket runs.
  d          flash_plain               the fork's `generate_batch` with both
                                       flags off, in eval mode.
  d_fixed_r  flash_plain_contiguous_r  `d` with R and T made contiguous before
                                       the two Triton affine kernels.

Noise. Every arm receives the same initial noise: one draw after
`torch.manual_seed(seed)` with the SO(3) SDE already on the device, saved to
`<prefix>_initial_noise.pt` and returned by a wrapper around each SDE's
`prior_sampling`. The wrapper calls the original sampler first, so random-number
consumption is unchanged, and records whether the native seeded draw already
equals the injected tensor. The per-step noise of the DPM sampler is zero in
every arm: `noise=0.0` in upstream `dpm.yaml` and in the harness, so
`t_hat == t` and the pre-step Euler-Maruyama update is multiplied by
`sqrt(|t_hat - t|) = 0`. Upstream still draws `z` there; the fork skips the
draw. The tool checks that this update is an exact no-op instead of injecting
noise that is then multiplied by zero. DPM-Solver-2 has no other stochastic
term. In train mode the dropout masks come from the CUDA generator after the
seeded prior draw, exactly as in an unmodified upstream call.

Captured tensors, all float32 on CPU, B=32 and L=76:

  pos, node_orientations          final sample, [B, L, 3] and [B, L, 3, 3].
  initial_pos, ...orientations    injected initial noise.
  first_step_score_t              raw score-model output of the first call of
                                  the first step, at t = max_t: pos [B, L, 3]
                                  and node_orientations (axis-angle) [B, L, 3].
  first_step_score_lambda         the second call of the first step, at the
                                  DPM midpoint t_lambda.
  first_step_state                the state after the first DPM step.

The taps copy into preallocated buffers in place, so a CUDA graph capture
records the copies and every replay refreshes them.

Usage, one process per source tree because both import as `bioemu`, then a
comparison that writes `summary.json`:

    PYTHONPATH=$OFFICIAL/src python -B bioemu_goldens.py run \
        --tree official --tree-root $OFFICIAL --out-dir $OUT [inputs]
    PYTHONPATH=$FLASH/src python -B bioemu_goldens.py run \
        --tree flash --tree-root $FLASH --out-dir $OUT [inputs]
    python -B bioemu_goldens.py compare --out-dir $OUT

`bioemu_goldens.sbatch` next to this file lists the inputs.
"""

import argparse
from collections.abc import Callable, Iterator, Mapping, Sequence
import contextlib
import dataclasses
import datetime
import hashlib
import importlib
import importlib.util
import json
import os
import pathlib
import socket
import subprocess
import sys
import time
import types
from typing import Any

import torch

_UBIQUITIN = (
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHL"
    "VLRLRGG"
)
_PAPER_CHECKPOINT_SHA256 = (
    "c2fa4e3094d5db1b465024b95d87b976272a2d9b63b7bbb8e97e98a192350fa2"
)
_MODEL_NAME = "bioemu-v1.1"
_MAX_T = 0.99
_EPS_T = 0.001
_NOISE = 0.0
_BENCHMARK_SCRIPT = "dev/benchmark_bioemu_throughput.py"
_ENV_KEYS = (
    "PYTHONPATH",
    "JAX_PLATFORM_NAME",
    "PYTHONDONTWRITEBYTECODE",
    "CUDA_VISIBLE_DEVICES",
    "TRITON_CACHE_DIR",
    "CUDA_CACHE_PATH",
    "SLURM_JOB_ID",
    "SLURM_JOB_NODELIST",
    "CC",
    "CXX",
)
# Command lines of `run_protein_size_scaling.py` for ubiquitin at batch 32.
_HARNESS_BASELINE_CLI = (
    "bioemu-official-benchmark/dev/benchmark_bioemu_throughput.py "
    "--sequence submission/data/protein_size_scaling/inputs/bioemu/"
    "ubiquitin_76aa.fasta --cache-embeds-dir submission/data/"
    "protein_size_scaling/bioemu_embed_cache --cache-so3-dir "
    "flash-bioemu/profiles/so3_cache --batch-size 32 --steps 9 "
    "--warmup-runs 2 --timed-runs 5 --activation-steps none "
    "--jax-platform cpu"
)
_HARNESS_FLASH_CLI = (
    "flash-bioemu/dev/benchmark_bioemu_throughput.py "
    "--sequence submission/data/protein_size_scaling/inputs/bioemu/"
    "ubiquitin_76aa.fasta --cache-embeds-dir submission/data/"
    "protein_size_scaling/bioemu_embed_cache --cache-so3-dir "
    "flash-bioemu/profiles/so3_cache --batch-size 32 --steps 9 "
    "--warmup-runs 5 --timed-runs 5 --activation-steps none "
    "--jax-platform cpu --static-context-cache --cuda-graph-dpm"
)


@dataclasses.dataclass(frozen=True)
class _Arm:
    """One code path whose output is recorded.

    Attributes:
        name: Identifier used in the output file name.
        tree: Source tree the arm imports, "official" or "flash".
        needs_cuda: Whether the arm cannot run without a GPU.
        taps: Whether the first-step tensors are captured.
        description: What the arm calls.
        flags: Entry point, model mode and harness flags of the arm.
    """

    name: str
    tree: str
    needs_cuda: bool
    taps: bool
    description: str
    flags: Mapping[str, Any]


_ARMS: dict[str, _Arm] = {
    "a": _Arm(
        name="official_eval",
        tree="official",
        needs_cuda=False,
        taps=True,
        description=(
            "Official upstream bioemu.sample.generate_batch -> "
            "bioemu.denoiser.dpm_solver, score model in eval mode."
        ),
        flags={
            "entry": "bioemu.sample.generate_batch",
            "denoiser": "bioemu.shortcuts.dpm_solver from dpm.yaml, N=10",
            "model_mode": "eval",
            "grad_mode": "torch.no_grad (as bioemu.sample.main)",
        },
    ),
    "b": _Arm(
        name="official_train",
        tree="official",
        needs_cuda=False,
        taps=True,
        description=(
            "Same call as a with the score model in train mode, as upstream "
            "bioemu.sample.main runs (it never calls .eval())."
        ),
        flags={
            "entry": "bioemu.sample.generate_batch",
            "denoiser": "bioemu.shortcuts.dpm_solver from dpm.yaml, N=10",
            "model_mode": "train (dropout 0.1 from the released config)",
            "grad_mode": "torch.no_grad (as bioemu.sample.main)",
        },
    ),
    "a_harness": _Arm(
        name="official_harness_eval",
        tree="official",
        needs_cuda=False,
        taps=True,
        description=(
            "run_once of the official benchmark harness: the paper's "
            "baseline series."
        ),
        flags={
            "entry": f"{_BENCHMARK_SCRIPT}:run_once",
            "model_mode": "eval (the harness calls score_model.eval())",
            "grad_mode": "torch.inference_mode",
            "harness_cli": _HARNESS_BASELINE_CLI,
        },
    ),
    "c": _Arm(
        name="flash_accelerated",
        tree="flash",
        needs_cuda=True,
        taps=True,
        description=(
            "CudaGraphDpmRunner of the fork's harness with the static context "
            "cache: the paper's flash series."
        ),
        flags={
            "entry": f"{_BENCHMARK_SCRIPT}:CudaGraphDpmRunner.run",
            "static_context_cache": True,
            "cuda_graph_dpm": True,
            "model_mode": "eval (the harness calls score_model.eval())",
            "grad_mode": "torch.inference_mode",
            "harness_cli": _HARNESS_FLASH_CLI,
        },
    ),
    "c_eager": _Arm(
        name="flash_static_cache_eager",
        tree="flash",
        needs_cuda=False,
        taps=True,
        description=(
            "run_once of the fork's harness with the static context cache and "
            "no CUDA graph."
        ),
        flags={
            "entry": f"{_BENCHMARK_SCRIPT}:run_once",
            "static_context_cache": True,
            "cuda_graph_dpm": False,
            "model_mode": "eval",
            "grad_mode": "torch.inference_mode",
        },
    ),
    "c_sample": _Arm(
        name="flash_sample_accelerated",
        tree="flash",
        needs_cuda=True,
        taps=False,
        description=(
            "The fork's bioemu.sample.generate_batch with "
            "static_context_cache=True and cuda_graph_dpm=True, the entry "
            "point of the cryptic-pocket runs."
        ),
        flags={
            "entry": "bioemu.sample.generate_batch",
            "static_context_cache": True,
            "cuda_graph_dpm": True,
            "model_mode": "eval (generate_batch calls score_model.eval())",
            "grad_mode": "torch.no_grad outside, inference_mode inside",
        },
    ),
    "d": _Arm(
        name="flash_plain",
        tree="flash",
        needs_cuda=False,
        taps=True,
        description=(
            "The fork's bioemu.sample.generate_batch with both flags off "
            "(the fork's dpm_solver on a plain Batch), eval mode."
        ),
        flags={
            "entry": "bioemu.sample.generate_batch",
            "static_context_cache": False,
            "cuda_graph_dpm": False,
            "model_mode": (
                "eval (set by this tool; the fork's sample.main would leave "
                "train mode on this path)"
            ),
            "grad_mode": "torch.no_grad",
        },
    ),
    "d_fixed_r": _Arm(
        name="flash_plain_contiguous_r",
        tree="flash",
        needs_cuda=False,
        taps=True,
        description=(
            "d with rotation and translation passed as contiguous tensors to "
            "triton_apply_affine_qkv and triton_invert_affine_norm."
        ),
        flags={
            "entry": "bioemu.sample.generate_batch",
            "static_context_cache": False,
            "cuda_graph_dpm": False,
            "model_mode": "eval",
            "grad_mode": "torch.no_grad",
            "patch": "R.contiguous(), T.contiguous() before the kernels",
        },
    ),
}
_REQUIRED_PAIRS = (("a", "c"), ("a", "d"), ("a", "b"), ("c", "d"))
_DIAGNOSTIC_PAIRS = (
    ("a", "a_harness"),
    ("a", "c_eager"),
    ("c", "c_eager"),
    ("c", "c_sample"),
    ("a", "d_fixed_r"),
    ("d", "d_fixed_r"),
)
_METRIC_DEFINITIONS = {
    "max_abs": "max |other - reference| over all elements",
    "max_rel": "max_abs / max |reference|",
    "rel_l2": "||other - reference||_2 / ||reference||_2",
    "rmsd_nm": (
        "per sample sqrt(mean over residues of |x_other - x_ref|^2), no "
        "superposition; then mean and max over the batch"
    ),
    "kabsch_rmsd_nm": "the same after optimal rigid superposition",
    "geodesic_deg": (
        "rotation angle of R_ref^T R_other per residue, mean and max"
    ),
    "arm_physics": (
        "per arm: consecutive C-alpha distances, the fraction of samples "
        "with all of them below 0.45 nm (upstream's filter threshold), and "
        "the radius of gyration, all in nm"
    ),
    "units": "pos in nm (BioEmu native); 1 nm = 10 Angstrom",
}


@dataclasses.dataclass
class _Session:
    """State shared by the arms of one source tree.

    Attributes:
        tree: "official" or "flash".
        tree_root: Root of the source tree.
        args: Parsed command-line arguments.
        device: Device the arms run on.
        modules: Imported `bioemu` modules and the benchmark harness.
        sequence: Protein sequence.
        score_model: Score model, shared by all arms.
        sdes: SDEs keyed by field name.
        context: Context ChemGraph built from the cached embeddings.
        denoiser: Hydra-instantiated `dpm_solver` partial.
        denoiser_config: Configuration the denoiser was built from.
        initial: Injected initial noise, flat [N, 3] and [N, 3, 3].
        meta: Process-level metadata written to `meta_<tree>.json`.
    """

    tree: str
    tree_root: pathlib.Path
    args: argparse.Namespace
    device: torch.device
    modules: dict[str, types.ModuleType]
    sequence: str
    score_model: torch.nn.Module
    sdes: dict[str, Any]
    context: Any
    denoiser: Callable[..., Any]
    denoiser_config: dict[str, Any]
    initial: dict[str, torch.Tensor]
    meta: dict[str, Any]

    @property
    def batch_size(self) -> int:
        """Number of samples per run."""
        return int(self.args.batch_size)

    @property
    def length(self) -> int:
        """Number of residues."""
        return len(self.sequence)

    @property
    def steps(self) -> int:
        """Number of DPM steps."""
        return int(self.args.steps)


class _Tap:
    """Copies tensors of selected periodic calls into fixed buffers.

    A denoising run calls the score model `2 * steps` times and the DPM step
    `steps` times. The tap counts calls modulo that period and copies the
    tensors of the selected calls in place, so a CUDA graph capture records
    the copies and each replay refreshes the buffers.
    """

    def __init__(
        self,
        period: int,
        slots: Mapping[int, str],
        shapes: Mapping[str, tuple[int, ...]],
        device: torch.device,
    ) -> None:
        self._period = period
        self._slots = dict(slots)
        self.calls = 0
        self.buffers = {
            (label, field): torch.full(shape, float("nan"), device=device)
            for label in self._slots.values()
            for field, shape in shapes.items()
        }

    def record(self, tensors: Mapping[str, torch.Tensor]) -> None:
        """Copies `tensors` if the current call is a selected one.

        Args:
            tensors: Tensors of this call, keyed by field name.
        """
        label = self._slots.get(self.calls % self._period)
        self.calls += 1
        if label is None:
            return
        for field, value in tensors.items():
            buffer = self.buffers[(label, field)]
            buffer.copy_(value.detach().reshape(buffer.shape))

    def collect(self) -> dict[str, dict[str, torch.Tensor]]:
        """Returns CPU copies of the buffers, keyed by label and field.

        Raises:
            RuntimeError: If a buffer was never written.
        """
        collected: dict[str, dict[str, torch.Tensor]] = {}
        for (label, field), buffer in self.buffers.items():
            value = buffer.detach().to("cpu").clone()
            if not torch.isfinite(value).all():
                raise RuntimeError(f"tap {label}/{field} was not written")
            collected.setdefault(label, {})[field] = value
        return collected


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Capture golden BioEmu outputs (see module docstring)."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run the arms of one source tree.")
    run.add_argument("--tree", choices=("official", "flash"), required=True)
    run.add_argument("--tree-root", type=pathlib.Path, required=True)
    run.add_argument("--out-dir", type=pathlib.Path, required=True)
    run.add_argument("--checkpoint", type=pathlib.Path, required=True)
    run.add_argument("--model-config", type=pathlib.Path, required=True)
    run.add_argument("--fasta", type=pathlib.Path, required=True)
    run.add_argument("--embeds-dir", type=pathlib.Path, required=True)
    run.add_argument("--so3-cache-dir", type=pathlib.Path, required=True)
    run.add_argument("--batch-size", type=int, default=32)
    run.add_argument("--steps", type=int, default=9)
    run.add_argument("--seed", type=int, default=20260616)
    run.add_argument("--prefix", default="ubiquitin")
    run.add_argument(
        "--arms", nargs="*", help="Subset of arm keys (default: all)."
    )
    run.add_argument(
        "--require-gpu",
        default="RTX PRO 6000",
        help="Substring the GPU name must contain.",
    )
    run.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Plumbing check on CPU; skips the arms that need CUDA.",
    )
    compare = commands.add_parser("compare", help="Write summary.json.")
    compare.add_argument("--out-dir", type=pathlib.Path, required=True)
    compare.add_argument("--prefix", default="ubiquitin")
    return parser.parse_args(argv)


def _sha256(path: pathlib.Path) -> str:
    """Returns the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_fingerprint(tree_root: pathlib.Path) -> dict[str, Any]:
    """Hashes the Python sources the arms import from a tree.

    Args:
        tree_root: Root of the source tree.

    Returns:
        SHA-256 over `src/bioemu/**/*.py` and the harness script, and the
        number of files hashed.
    """
    files = sorted(
        path
        for path in (tree_root / "src" / "bioemu").rglob("*.py")
        if "__pycache__" not in path.parts
    )
    harness = tree_root / _BENCHMARK_SCRIPT
    if harness.is_file():
        files.append(harness)
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(tree_root)).encode())
        digest.update(path.read_bytes())
    return {"sha256": digest.hexdigest(), "files": len(files)}


def _git_info(tree_root: pathlib.Path) -> dict[str, str]:
    """Reads the commit and the working-tree state without writing.

    Args:
        tree_root: Root of a git checkout.

    Returns:
        HEAD, its subject, tracked changes and untracked files under src.
    """

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(tree_root), *args],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() or result.stderr.strip()

    return {
        "head": git("rev-parse", "HEAD"),
        "subject": git("log", "-1", "--format=%s"),
        "tracked_changes": git("status", "--porcelain", "--untracked-files=no"),
        "src_status": git("status", "--porcelain", "--", "src"),
    }


def _package_versions() -> dict[str, str]:
    """Returns versions of the packages that affect the numbers."""
    versions = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": str(torch.version.cuda),
        "cudnn": str(torch.backends.cudnn.version()),
    }
    for name in ("numpy", "torch_geometric", "triton", "hydra", "scipy"):
        try:
            versions[name] = importlib.import_module(name).__version__
        except Exception as error:
            versions[name] = f"unavailable: {error}"
    return versions


def _gpu_info() -> dict[str, Any]:
    """Returns the GPU name, capability and driver, if a GPU is present."""
    if not torch.cuda.is_available():
        return {"name": None}
    props = torch.cuda.get_device_properties(0)
    smi = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,uuid",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "name": props.name,
        "capability": f"{props.major}.{props.minor}",
        "total_memory_bytes": props.total_memory,
        "nvidia_smi": smi.stdout.strip(),
    }


def _numerics_flags() -> dict[str, Any]:
    """Returns the global PyTorch settings that change float results."""
    return {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "deterministic": torch.are_deterministic_algorithms_enabled(),
    }


def _load_module(path: pathlib.Path, name: str) -> types.ModuleType:
    """Imports a Python file as a module without writing bytecode.

    Args:
        path: File to import.
        name: Module name to give it.

    Returns:
        The executed module.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _import_tree(
    tree: str, tree_root: pathlib.Path
) -> dict[str, types.ModuleType]:
    """Imports `bioemu` and checks that it comes from `tree_root`.

    Args:
        tree: "official" or "flash".
        tree_root: Root of the source tree on PYTHONPATH.

    Returns:
        The modules the arms use, keyed by short name.

    Raises:
        RuntimeError: If `bioemu` resolves to another installation.
    """
    bioemu = importlib.import_module("bioemu")
    origin = pathlib.Path(bioemu.__file__).resolve()
    src = (tree_root / "src").resolve()
    if src not in origin.parents:
        raise RuntimeError(f"bioemu imported from {origin}, expected {src}")
    modules = {"bioemu": bioemu}
    for name in ("denoiser", "sample", "model_utils", "seq_io"):
        modules[name] = importlib.import_module(f"bioemu.{name}")
    modules["structure_module"] = importlib.import_module(
        "bioemu.structure_module"
    )
    if tree == "official":
        modules["benchmark"] = _load_module(
            tree_root / _BENCHMARK_SCRIPT, "golden_official_benchmark"
        )
    else:
        # The module object the fork's sample.py uses for accelerated runs.
        modules["benchmark"] = modules["sample"]._benchmark_module()
    origin = pathlib.Path(modules["benchmark"].__file__).resolve()
    if tree_root.resolve() not in origin.parents:
        raise RuntimeError(f"harness imported from {origin}")
    return modules


@contextlib.contextmanager
def _patch_attr(obj: Any, name: str, value: Any) -> Iterator[None]:
    """Temporarily sets an attribute and restores the previous state.

    Args:
        obj: Object, class or module to patch.
        name: Attribute name.
        value: Temporary value.

    Yields:
        None, while the attribute is patched.
    """
    own = vars(obj)
    had = name in own
    previous = own.get(name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        if had:
            setattr(obj, name, previous)
        else:
            delattr(obj, name)


@contextlib.contextmanager
def _inject_prior(
    sdes: Mapping[str, Any], initial: Mapping[str, torch.Tensor]
) -> Iterator[list[dict[str, Any]]]:
    """Makes both SDE priors return the stored initial noise.

    The original sampler still runs first, so random-number consumption is
    the same as without the patch.

    Args:
        sdes: SDEs keyed by field name.
        initial: Initial noise keyed by field name.

    Yields:
        A log with one entry per prior draw.
    """
    log: list[dict[str, Any]] = []
    with contextlib.ExitStack() as stack:
        for field in ("pos", "node_orientations"):
            sde = sdes[field]
            original = sde.prior_sampling
            value = initial[field]

            def inject(
                shape: Any,
                device: Any = None,
                *,
                _field: str = field,
                _original: Callable[..., torch.Tensor] = original,
                _value: torch.Tensor = value,
            ) -> torch.Tensor:
                drawn = _original(shape, device=device)
                if tuple(drawn.shape) != tuple(_value.shape):
                    raise ValueError(
                        f"{_field} prior shape {tuple(drawn.shape)}, "
                        f"stored {tuple(_value.shape)}"
                    )
                target = _value.to(drawn.device)
                log.append(
                    {
                        "field": _field,
                        "device": str(drawn.device),
                        "seeded_draw_equals_injected": bool(
                            torch.equal(drawn, target)
                        ),
                        "max_abs_seeded_minus_injected": float(
                            (drawn - target).abs().max()
                        ),
                    }
                )
                return target.clone()

            stack.enter_context(_patch_attr(sde, "prior_sampling", inject))
        yield log


@contextlib.contextmanager
def _watch_pre_step_noise(
    denoiser_module: types.ModuleType,
) -> Iterator[dict[str, Any]]:
    """Measures the pre-step noise update of the DPM sampler.

    Args:
        denoiser_module: The tree's `bioemu.denoiser`.

    Yields:
        Call count and the largest |x_hat - x| and |dt| seen.
    """
    stats: dict[str, Any] = {
        "calls": 0,
        "max_abs_update": 0.0,
        "max_abs_dt": 0.0,
    }
    predictor = denoiser_module.EulerMaruyamaPredictor
    original = predictor.forward_sde_step

    def watched(self: Any, *args: Any, **kwargs: Any) -> Any:
        x = kwargs["x"] if "x" in kwargs else args[0]
        dt = kwargs["dt"] if "dt" in kwargs else args[2]
        sample, mean = original(self, *args, **kwargs)
        stats["calls"] += 1
        stats["max_abs_update"] = max(
            stats["max_abs_update"], float((sample - x).abs().max())
        )
        stats["max_abs_dt"] = max(
            stats["max_abs_dt"], float(torch.as_tensor(dt).abs().max())
        )
        return sample, mean

    with _patch_attr(predictor, "forward_sde_step", watched):
        yield stats


@contextlib.contextmanager
def _watch_affine_kernels(
    structure_module: types.ModuleType, make_contiguous: bool
) -> Iterator[dict[str, Any]]:
    """Records the layout of R and T at the fork's Triton affine kernels.

    The kernels index R as a dense [B, L, 3, 3] array without strides, so a
    non-contiguous R is read as its transpose.

    Args:
        structure_module: The fork's `bioemu.structure_module`.
        make_contiguous: Pass contiguous copies of R and T to the kernels.

    Yields:
        Call counts, counts of non-contiguous R and T, and the R strides.
    """
    stats: dict[str, Any] = {
        "kernels_available": False,
        "calls": {},
        "noncontiguous_rotation_calls": 0,
        "noncontiguous_translation_calls": 0,
        "rotation_strides": [],
        "made_contiguous": make_contiguous,
    }
    names = ("triton_apply_affine_qkv", "triton_invert_affine_norm")
    with contextlib.ExitStack() as stack:
        for name in names:
            original = getattr(structure_module, name, None)
            if original is None:
                continue
            stats["kernels_available"] = True
            stats["calls"][name] = 0

            def wrapped(
                *args: torch.Tensor,
                _name: str = name,
                _original: Callable[..., Any] = original,
            ) -> Any:
                *points, translation, rotation = args
                stats["calls"][_name] += 1
                if not rotation.is_contiguous():
                    stats["noncontiguous_rotation_calls"] += 1
                if not translation.is_contiguous():
                    stats["noncontiguous_translation_calls"] += 1
                strides = list(rotation.stride())
                if strides not in stats["rotation_strides"]:
                    stats["rotation_strides"].append(strides)
                if make_contiguous:
                    translation = translation.contiguous()
                    rotation = rotation.contiguous()
                return _original(*points, translation, rotation)

            stack.enter_context(_patch_attr(structure_module, name, wrapped))
        yield stats


@contextlib.contextmanager
def _taps(
    session: _Session, step_owner: types.ModuleType
) -> Iterator[tuple[_Tap, _Tap]]:
    """Installs the first-step taps on the score model and the DPM step.

    Args:
        session: Current session.
        step_owner: Module whose global `second_order_step_dpmsolver` the
            sampler of this arm calls.

    Yields:
        The score-model tap and the step tap.
    """
    b, n, steps = session.batch_size, session.length, session.steps
    score_tap = _Tap(
        period=2 * steps,
        slots={0: "first_step_score_t", 1: "first_step_score_lambda"},
        shapes={"pos": (b, n, 3), "node_orientations": (b, n, 3)},
        device=session.device,
    )
    step_tap = _Tap(
        period=steps,
        slots={0: "first_step_state"},
        shapes={"pos": (b, n, 3), "node_orientations": (b, n, 3, 3)},
        device=session.device,
    )

    def hook(module: torch.nn.Module, inputs: Any, output: Any) -> None:
        del module, inputs
        score_tap.record(
            {
                "pos": output["pos"],
                "node_orientations": output["node_orientations"],
            }
        )

    original = step_owner.second_order_step_dpmsolver

    def step(*args: Any, **kwargs: Any) -> Any:
        batch = original(*args, **kwargs)
        step_tap.record(
            {"pos": batch.pos, "node_orientations": batch.node_orientations}
        )
        return batch

    handle = session.score_model.register_forward_hook(hook)
    try:
        with _patch_attr(step_owner, "second_order_step_dpmsolver", step):
            yield score_tap, step_tap
    finally:
        handle.remove()


def _to_cpu(x: torch.Tensor) -> torch.Tensor:
    """Returns a detached CPU copy that is a normal (non-inference) tensor."""
    return x.detach().to("cpu").clone()


def _frames(batch: Any, b: int, n: int) -> dict[str, torch.Tensor]:
    """Reshapes a sampled ChemGraph batch to [B, L, ...] CPU tensors."""
    return {
        "pos": _to_cpu(batch.pos).reshape(b, n, 3),
        "node_orientations": _to_cpu(batch.node_orientations).reshape(
            b, n, 3, 3
        ),
    }


def _sync() -> None:
    """Waits for queued GPU work, if any."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _run_generate_batch(
    session: _Session, *, train: bool, accelerated: bool
) -> dict[str, torch.Tensor]:
    """Runs the tree's `bioemu.sample.generate_batch` once.

    Args:
        session: Current session.
        train: Put the score model in train mode, else eval mode.
        accelerated: For the fork, set both acceleration flags.

    Returns:
        Final pos [B, L, 3] and node_orientations [B, L, 3, 3].
    """
    kwargs: dict[str, Any] = {
        "score_model": session.score_model,
        "sequence": session.sequence,
        "sdes": session.sdes,
        "batch_size": session.batch_size,
        "seed": session.args.seed,
        "denoiser": session.denoiser,
        "cache_embeds_dir": str(session.args.embeds_dir),
    }
    if session.tree == "flash":
        kwargs.update(
            static_context_cache=accelerated,
            cuda_graph_dpm=accelerated,
            # What the fork's sample.main passes.
            denoiser_settings=dict(getattr(session.denoiser, "keywords", {})),
        )
    session.score_model.train(train)
    with torch.no_grad():
        frames = session.modules["sample"].generate_batch(**kwargs)
    _sync()
    return {key: _to_cpu(value) for key, value in frames.items()}


def _run_benchmark_once(
    session: _Session, *, static_context_cache: bool | None
) -> dict[str, torch.Tensor]:
    """Runs `run_once` of the tree's benchmark harness.

    Args:
        session: Current session.
        static_context_cache: Value for the fork's flag; None for the
            official harness, which has no such argument.

    Returns:
        Final pos [B, L, 3] and node_orientations [B, L, 3, 3].
    """
    kwargs: dict[str, Any] = {
        "context": session.context,
        "batch_size": session.batch_size,
        "sdes": session.sdes,
        "score_model": session.score_model,
        "device": session.device,
        "seed": session.args.seed,
        "steps": session.steps,
        "max_t": _MAX_T,
        "eps_t": _EPS_T,
        "noise": _NOISE,
        "capture_steps": set(),
    }
    if static_context_cache is not None:
        kwargs["static_context_cache"] = static_context_cache
    session.score_model.eval()
    _, final, _ = session.modules["benchmark"].run_once(**kwargs)
    _sync()
    return _frames(final, session.batch_size, session.length)


def _build_graph_runner(session: _Session) -> Any:
    """Builds the fork's `CudaGraphDpmRunner` as the harness does.

    Args:
        session: Current session.

    Returns:
        The runner, with its graph captured.
    """
    session.score_model.eval()
    return session.modules["benchmark"].CudaGraphDpmRunner(
        context=session.context,
        batch_size=session.batch_size,
        sdes=session.sdes,
        score_model=session.score_model,
        device=session.device,
        seed=session.args.seed,
        steps=session.steps,
        max_t=_MAX_T,
        eps_t=_EPS_T,
        noise=_NOISE,
        static_context_cache=True,
    )


def _replay(session: _Session, runner: Any) -> dict[str, torch.Tensor]:
    """Replays a captured runner and copies its static outputs out.

    Args:
        session: Current session.
        runner: A `CudaGraphDpmRunner`.

    Returns:
        Final pos [B, L, 3] and node_orientations [B, L, 3, 3].
    """
    _, final, _ = runner.run(seed=session.args.seed)
    _sync()
    return _frames(final, session.batch_size, session.length)


def _max_abs(x: torch.Tensor, y: torch.Tensor) -> float:
    """Returns max |x - y| in float64."""
    return float((x.double() - y.double()).abs().max())


def _run_arm(session: _Session, key: str) -> dict[str, Any]:
    """Runs one arm with prior injection, noise check and taps.

    Args:
        session: Current session.
        key: Arm key in `_ARMS`.

    Returns:
        The record saved for the arm.
    """
    arm = _ARMS[key]
    modules = session.modules
    extra: dict[str, Any] = {}
    start = time.perf_counter()
    with contextlib.ExitStack() as stack:
        prior_log = stack.enter_context(
            _inject_prior(session.sdes, session.initial)
        )
        noise_stats = stack.enter_context(
            _watch_pre_step_noise(modules["denoiser"])
        )
        kernel_stats = None
        if session.tree == "flash":
            kernel_stats = stack.enter_context(
                _watch_affine_kernels(
                    modules["structure_module"],
                    make_contiguous=key == "d_fixed_r",
                )
            )
        if key == "c":
            # Graph without taps first: the taps add copy kernels to the
            # graph, and this checks that they leave the result unchanged.
            untapped = _replay(session, _build_graph_runner(session))
        taps = None
        if arm.taps:
            owner = (
                modules["benchmark"]
                if key in ("a_harness", "c", "c_eager")
                else modules["denoiser"]
            )
            taps = stack.enter_context(_taps(session, owner))
        if key in ("a", "b"):
            out = _run_generate_batch(
                session, train=key == "b", accelerated=False
            )
            training = session.score_model.training
            tapped = _collect_taps(taps)
            repeat = _run_generate_batch(
                session, train=key == "b", accelerated=False
            )
            extra["repeat_max_abs_pos"] = _max_abs(out["pos"], repeat["pos"])
        elif key == "a_harness":
            out = _run_benchmark_once(session, static_context_cache=None)
            training = session.score_model.training
        elif key == "c":
            runner = _build_graph_runner(session)
            out = _replay(session, runner)
            training = session.score_model.training
            tapped = _collect_taps(taps)
            repeat = _replay(session, runner)
            extra["repeat_max_abs_pos"] = _max_abs(out["pos"], repeat["pos"])
            extra["untapped_graph_max_abs_pos"] = _max_abs(
                out["pos"], untapped["pos"]
            )
            extra["untapped_graph_max_abs_node_orientations"] = _max_abs(
                out["node_orientations"], untapped["node_orientations"]
            )
        elif key == "c_eager":
            out = _run_benchmark_once(session, static_context_cache=True)
            training = session.score_model.training
        elif key == "c_sample":
            out = _run_generate_batch(session, train=False, accelerated=True)
            training = session.score_model.training
        elif key in ("d", "d_fixed_r"):
            out = _run_generate_batch(session, train=False, accelerated=False)
            training = session.score_model.training
        else:
            raise KeyError(key)
        if key not in ("a", "b", "c"):
            tapped = _collect_taps(taps)
    record: dict[str, Any] = {
        "arm": key,
        "name": arm.name,
        "tree": session.tree,
        "pos": out["pos"],
        "node_orientations": out["node_orientations"],
        "initial_pos": _to_cpu(session.initial["pos"]).reshape(
            session.batch_size, session.length, 3
        ),
        "initial_node_orientations": _to_cpu(
            session.initial["node_orientations"]
        ).reshape(session.batch_size, session.length, 3, 3),
    }
    record.update(tapped)
    meta = {
        "arm": key,
        "name": arm.name,
        "tree": session.tree,
        "description": arm.description,
        "flags": dict(arm.flags),
        "score_model_training_after_run": training,
        "prior_injection": prior_log,
        "pre_step_noise": noise_stats,
        "affine_kernels": kernel_stats,
        "first_step_tensors": sorted(tapped),
        "wall_s": time.perf_counter() - start,
        **extra,
    }
    record["meta_json"] = json.dumps(meta)
    return record


def _collect_taps(
    taps: tuple[_Tap, _Tap] | None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Returns the buffers of the taps, or an empty dict without taps."""
    if taps is None:
        return {}
    score_tap, step_tap = taps
    return {**score_tap.collect(), **step_tap.collect()}


def _seeded_prior(
    sdes: Mapping[str, Any], n: int, seed: int, device: torch.device
) -> dict[str, torch.Tensor]:
    """Draws the initial noise as every arm draws it after seeding.

    Args:
        sdes: SDEs keyed by field name; the SO(3) SDE is on `device`.
        n: Number of nodes, batch size times residues.
        seed: Seed passed to `torch.manual_seed`.
        device: Device of the draw.

    Returns:
        pos [n, 3] and node_orientations [n, 3, 3].
    """
    torch.manual_seed(seed)
    pos = sdes["pos"].prior_sampling(torch.Size((n, 3)), device=device)
    rot = sdes["node_orientations"].prior_sampling(
        torch.Size((n, 3, 3)), device=device
    )
    return {"pos": pos, "node_orientations": rot}


def _initial_noise(
    args: argparse.Namespace,
    sdes: Mapping[str, Any],
    n: int,
    device: torch.device,
    tree: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Loads the shared initial noise, creating it on first use.

    Args:
        args: Parsed arguments.
        sdes: SDEs keyed by field name.
        n: Number of nodes.
        device: Device of the draw.
        tree: Tree of the current process.

    Returns:
        The noise on `device` and a record of how it was obtained.
    """
    path = args.out_dir / f"{args.prefix}_initial_noise.pt"
    drawn = _seeded_prior(sdes, n, args.seed, device)
    if path.exists():
        stored = torch.load(path, map_location="cpu", weights_only=True)
        if stored["seed"] != args.seed or stored["pos"].shape[0] != n:
            raise ValueError(f"{path} was made for another seed or size")
        initial = {
            "pos": stored["pos"].to(device),
            "node_orientations": stored["node_orientations"].to(device),
        }
        info = {"path": str(path), "created_by": stored["created_by"]}
    else:
        initial = drawn
        torch.save(
            {
                "pos": _to_cpu(drawn["pos"]),
                "node_orientations": _to_cpu(drawn["node_orientations"]),
                "seed": args.seed,
                "created_by": tree,
                "device": str(device),
            },
            path,
        )
        info = {"path": str(path), "created_by": tree}
    info["this_tree_draw_equals_stored"] = all(
        torch.equal(drawn[key], initial[key]) for key in initial
    )
    return initial, info


def _build_session(args: argparse.Namespace) -> _Session:
    """Imports the tree, loads model, SDEs, context and noise.

    Args:
        args: Parsed `run` arguments.

    Returns:
        The session shared by the arms.
    """
    # Imported here so that `bioemu` is resolved and checked first.
    import hydra
    import yaml

    tree_root = args.tree_root.resolve()
    modules = _import_tree(args.tree, tree_root)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        name = torch.cuda.get_device_name(0)
        if args.require_gpu and args.require_gpu not in name:
            raise RuntimeError(f"GPU {name!r} lacks {args.require_gpu!r}")
    elif args.allow_cpu:
        device = torch.device("cpu")
    else:
        raise RuntimeError("no GPU; pass --allow-cpu for a plumbing check")

    sequence = modules["seq_io"].parse_sequence(str(args.fasta))
    if sequence != _UBIQUITIN:
        raise ValueError(f"{args.fasta} is not the paper's ubiquitin")
    seq_sha = hashlib.sha256(sequence.encode()).hexdigest()
    embeds = {
        kind: args.embeds_dir / f"{seq_sha}_{kind}.npy"
        for kind in ("single", "pair")
    }
    for path in embeds.values():
        # A miss would start ColabFold and write a new cache entry.
        if not path.is_file():
            raise FileNotFoundError(path)

    checkpoint_sha = _sha256(args.checkpoint)
    if checkpoint_sha != _PAPER_CHECKPOINT_SHA256:
        raise ValueError(f"checkpoint sha256 {checkpoint_sha} is not v1.1")
    model_utils = modules["model_utils"]
    score_model = model_utils.load_model(args.checkpoint, args.model_config)
    score_model.to(device)
    sdes = model_utils.load_sdes(
        model_config_path=args.model_config,
        cache_so3_dir=str(args.so3_cache_dir),
    )
    # dpm_solver and the harness move it themselves; doing it first makes
    # every prior draw come from the device generator.
    sdes["node_orientations"].to(device)
    context = modules["sample"].get_context_chemgraph(
        sequence=sequence, cache_embeds_dir=str(args.embeds_dir)
    )

    config_path = modules["sample"].DEFAULT_DENOISER_CONFIG_DIR / "dpm.yaml"
    with config_path.open() as handle:
        denoiser_config = yaml.safe_load(handle)
    released = dict(denoiser_config)
    expected = {"max_t": _MAX_T, "eps_t": _EPS_T, "noise": _NOISE}
    for field, value in expected.items():
        if denoiser_config[field] != value:
            raise ValueError(f"dpm.yaml {field}={denoiser_config[field]}")
    denoiser_config["N"] = args.steps + 1
    denoiser = hydra.utils.instantiate(denoiser_config)
    # Silence the progress bar; it has no effect on the numbers.
    modules["denoiser"].tqdm = lambda iterable, **_: iterable

    n = args.batch_size * len(sequence)
    initial, noise_info = _initial_noise(args, sdes, n, device, args.tree)
    with args.model_config.open() as handle:
        model_config = yaml.safe_load(handle)
    dropout = sorted(
        {
            module.p
            for module in score_model.modules()
            if isinstance(module, torch.nn.Dropout)
        }
    )
    meta = {
        "tree": args.tree,
        "tree_root": str(tree_root),
        "git": _git_info(tree_root),
        "code_fingerprint": _tree_fingerprint(tree_root),
        "bioemu_file": modules["bioemu"].__file__,
        "bioemu_version": getattr(modules["bioemu"], "__version__", None),
        "harness_file": modules["benchmark"].__file__,
        "versions": _package_versions(),
        "gpu": _gpu_info(),
        "numerics": _numerics_flags(),
        "hostname": socket.gethostname(),
        "env": {key: os.environ.get(key) for key in _ENV_KEYS},
        "argv": sys.argv,
        "device": str(device),
        "model_name": _MODEL_NAME,
        "checkpoint": {"path": str(args.checkpoint), "sha256": checkpoint_sha},
        "model_config": {
            "path": str(args.model_config),
            "sha256": _sha256(args.model_config),
            "score_model": model_config["score_model"],
        },
        "dropout_p_in_model": dropout,
        "sequence": sequence,
        "fasta": {"path": str(args.fasta), "sha256": _sha256(args.fasta)},
        "embeds": {
            kind: {"path": str(path), "sha256": _sha256(path)}
            for kind, path in embeds.items()
        },
        "so3_cache": {
            path.name: _sha256(path)
            for path in sorted(args.so3_cache_dir.glob("*.npz"))
        },
        "denoiser": {
            "config_path": str(config_path),
            "released_config": released,
            "used_config": denoiser_config,
            "steps": args.steps,
            "score_model_calls_per_run": 2 * args.steps,
        },
        "batch_size": args.batch_size,
        "seed": args.seed,
        "initial_noise": noise_info,
    }
    return _Session(
        tree=args.tree,
        tree_root=tree_root,
        args=args,
        device=device,
        modules=modules,
        sequence=sequence,
        score_model=score_model,
        sdes=sdes,
        context=context,
        denoiser=denoiser,
        denoiser_config=denoiser_config,
        initial=initial,
        meta=meta,
    )


def _run(args: argparse.Namespace) -> None:
    """Runs and saves the arms of one source tree."""
    args.out_dir.mkdir(parents=True, exist_ok=True)
    session = _build_session(args)
    keys = [key for key, arm in _ARMS.items() if arm.tree == args.tree]
    if args.arms:
        keys = [key for key in keys if key in args.arms]
    arm_meta: dict[str, Any] = {}
    for key in keys:
        arm = _ARMS[key]
        if arm.needs_cuda and session.device.type != "cuda":
            arm_meta[key] = {"skipped": "needs CUDA"}
            print(f"SKIP {key} ({arm.name}): needs CUDA", flush=True)
            continue
        print(f"RUN {key} ({arm.name})", flush=True)
        record = _run_arm(session, key)
        path = args.out_dir / f"{args.prefix}_{arm.name}.pt"
        torch.save(record, path)
        arm_meta[key] = json.loads(record["meta_json"])
        arm_meta[key]["file"] = str(path)
        print(f"WROTE {path}", flush=True)
    session.meta["arms"] = arm_meta
    meta_path = args.out_dir / f"{args.prefix}_meta_{args.tree}.json"
    meta_path.write_text(json.dumps(session.meta, indent=2) + "\n")
    print(f"WROTE {meta_path}", flush=True)


def _tensor_diff(ref: torch.Tensor, other: torch.Tensor) -> dict[str, float]:
    """Returns max-abs, max-rel and relative L2 differences.

    Args:
        ref: Reference tensor.
        other: Tensor compared with `ref`, same shape.

    Returns:
        The metrics in `_METRIC_DEFINITIONS`, as floats.
    """
    r, o = ref.double(), other.double()
    delta = (o - r).abs()
    scale = float(r.abs().max())
    return {
        "max_abs": float(delta.max()),
        "max_rel": float(delta.max()) / scale if scale else float("nan"),
        "rel_l2": float(delta.norm() / r.norm()),
        "ref_max_abs": scale,
        "nonfinite": int((~torch.isfinite(o)).sum()),
    }


def _kabsch_rmsd(ref: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """Returns the per-sample RMSD after optimal rigid superposition.

    Args:
        ref: Reference coordinates [B, L, 3].
        other: Coordinates [B, L, 3] superposed onto `ref`.

    Returns:
        RMSD per sample [B].
    """
    p = ref - ref.mean(dim=-2, keepdim=True)
    q = other - other.mean(dim=-2, keepdim=True)
    u, _, vh = torch.linalg.svd(q.mT @ p)
    v = vh.mT
    sign = torch.sign(torch.linalg.det(v @ u.mT))
    diag = torch.ones((*sign.shape, 3), dtype=p.dtype)
    diag[..., -1] = sign
    rotation = v @ torch.diag_embed(diag) @ u.mT
    aligned = q @ rotation.mT
    return (p - aligned).square().sum(-1).mean(-1).sqrt()


def _position_metrics(ref: torch.Tensor, other: torch.Tensor) -> dict[str, Any]:
    """Compares final positions [B, L, 3].

    Args:
        ref: Reference positions.
        other: Compared positions.

    Returns:
        RMSD per sample with and without superposition, and max |d pos|.
    """
    r, o = ref.double(), other.double()
    rmsd = (o - r).square().sum(-1).mean(-1).sqrt()
    kabsch = _kabsch_rmsd(r, o)
    return {
        "rmsd_nm_mean": float(rmsd.mean()),
        "rmsd_nm_max": float(rmsd.max()),
        "rmsd_nm_per_sample": [float(value) for value in rmsd],
        "kabsch_rmsd_nm_mean": float(kabsch.mean()),
        "kabsch_rmsd_nm_max": float(kabsch.max()),
        "max_abs_nm": float((o - r).abs().max()),
        "nonfinite": int((~torch.isfinite(o)).sum()),
    }


def _orientation_metrics(
    ref: torch.Tensor, other: torch.Tensor
) -> dict[str, float]:
    """Compares final orientations [B, L, 3, 3].

    Args:
        ref: Reference rotation matrices.
        other: Compared rotation matrices.

    Returns:
        Max |d R| and the geodesic angle between the rotations.
    """
    r, o = ref.double(), other.double()
    relative = r.mT @ o
    # atan2 of the skew and trace parts stays accurate near zero, where
    # arccos of the trace alone is limited by float32 orthogonality errors.
    skew = torch.stack(
        [
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ],
        dim=-1,
    )
    sin = 0.5 * skew.norm(dim=-1)
    cos = 0.5 * (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1)
    angle = torch.rad2deg(torch.atan2(sin, cos))
    return {
        "max_abs": float((o - r).abs().max()),
        "geodesic_deg_mean": float(angle.mean()),
        "geodesic_deg_max": float(angle.max()),
    }


def _physics(pos: torch.Tensor) -> dict[str, float]:
    """Returns sanity numbers of C-alpha traces [B, L, 3] in nm.

    Args:
        pos: Final positions of one arm.

    Returns:
        Consecutive C-alpha distances, the fraction of samples whose chain
        passes upstream's 4.5 Angstrom C-alpha filter, and the radius of
        gyration.
    """
    x = pos.double()
    bond = (x[:, 1:] - x[:, :-1]).norm(dim=-1)
    rg = (x - x.mean(dim=1, keepdim=True)).square().sum(-1).mean(-1).sqrt()
    return {
        "ca_ca_nm_mean": float(bond.mean()),
        "ca_ca_nm_min": float(bond.min()),
        "ca_ca_nm_max": float(bond.max()),
        "frac_samples_ca_chain_below_0p45nm": float(
            (bond.max(dim=1).values < 0.45).double().mean()
        ),
        "rg_nm_mean": float(rg.mean()),
        "rg_nm_min": float(rg.min()),
        "rg_nm_max": float(rg.max()),
    }


def _compare_pair(
    ref: Mapping[str, Any], other: Mapping[str, Any]
) -> dict[str, Any]:
    """Computes every metric of one pair of arm records.

    Args:
        ref: Reference arm record.
        other: Compared arm record.

    Returns:
        Metrics for the first-step tensors, final positions and orientations.
    """
    result: dict[str, Any] = {
        "reference": ref["name"],
        "other": other["name"],
        "same_initial_noise": bool(
            torch.equal(ref["initial_pos"], other["initial_pos"])
            and torch.equal(
                ref["initial_node_orientations"],
                other["initial_node_orientations"],
            )
        ),
    }
    for label in (
        "first_step_score_t",
        "first_step_score_lambda",
        "first_step_state",
    ):
        if label in ref and label in other:
            result[label] = {
                field: _tensor_diff(ref[label][field], other[label][field])
                for field in ref[label]
            }
    result["final_positions"] = _position_metrics(ref["pos"], other["pos"])
    result["final_orientations"] = _orientation_metrics(
        ref["node_orientations"], other["node_orientations"]
    )
    return result


def _self_test_kabsch() -> float:
    """Returns the superposed RMSD of a rigidly moved copy (expect ~0)."""
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(2, 76, 3, generator=generator, dtype=torch.float64)
    q, _ = torch.linalg.qr(torch.randn(3, 3, generator=generator).double())
    q = q * torch.sign(torch.linalg.det(q))
    moved = x @ q.mT + torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    return float(_kabsch_rmsd(x, moved).max())


def _table_row(key: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Flattens the headline numbers of one pair for the table."""
    row: dict[str, Any] = {"pair": key}
    score = metrics.get("first_step_score_t")
    if score:
        for field, short in (("pos", "pos"), ("node_orientations", "rot")):
            row[f"score_{short}_max_abs"] = score[field]["max_abs"]
            row[f"score_{short}_max_rel"] = score[field]["max_rel"]
    final = metrics["final_positions"]
    row["rmsd_nm_mean"] = final["rmsd_nm_mean"]
    row["rmsd_nm_max"] = final["rmsd_nm_max"]
    row["kabsch_rmsd_nm_mean"] = final["kabsch_rmsd_nm_mean"]
    row["max_abs_pos_nm"] = final["max_abs_nm"]
    row["rot_geodesic_deg_max"] = metrics["final_orientations"][
        "geodesic_deg_max"
    ]
    return row


def _compare(args: argparse.Namespace) -> None:
    """Writes `summary.json` from the saved arms and process metadata."""
    out = args.out_dir
    metas = {}
    for tree in ("official", "flash"):
        path = out / f"{args.prefix}_meta_{tree}.json"
        if path.exists():
            metas[tree] = json.loads(path.read_text())
    records = {}
    for key, arm in _ARMS.items():
        path = out / f"{args.prefix}_{arm.name}.pt"
        if path.exists():
            records[key] = torch.load(path, weights_only=True)
    pairwise: dict[str, Any] = {}
    for kind, pairs in (
        ("required", _REQUIRED_PAIRS),
        ("diagnostic", _DIAGNOSTIC_PAIRS),
    ):
        for ref_key, other_key in pairs:
            if ref_key in records and other_key in records:
                metrics = _compare_pair(records[ref_key], records[other_key])
                metrics["kind"] = kind
                pairwise[f"{ref_key}_vs_{other_key}"] = metrics
    arms = {}
    for key, record in records.items():
        meta = json.loads(record["meta_json"])
        meta["file"] = str(out / f"{args.prefix}_{record['name']}.pt")
        arms[key] = meta
    any_meta = next(iter(metas.values()), {})

    def rmsd(pair: str) -> float | None:
        if pair not in pairwise:
            return None
        return pairwise[pair]["final_positions"]["rmsd_nm_mean"]

    def kernel_stats(key: str) -> Any:
        return arms.get(key, {}).get("affine_kernels")

    summary = {
        "purpose": (
            "Golden BioEmu outputs of the official tree and the FlashBioEmu "
            "fork, captured before the fork becomes an adapter over PyPI "
            "bioemu==1.4.1."
        ),
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "system": "ubiquitin",
        "sequence": any_meta.get("sequence"),
        "model_name": _MODEL_NAME,
        "batch_size": any_meta.get("batch_size"),
        "seed": any_meta.get("seed"),
        "sampler": {
            "name": "DPM-Solver-2 (bioemu dpm_solver)",
            "steps": any_meta.get("denoiser", {}).get("steps"),
            "N": any_meta.get("denoiser", {}).get("used_config", {}).get("N"),
            "max_t": _MAX_T,
            "eps_t": _EPS_T,
            "noise": _NOISE,
            "score_model_calls_per_run": any_meta.get("denoiser", {}).get(
                "score_model_calls_per_run"
            ),
        },
        "noise_handling": (
            "Initial noise: one draw after torch.manual_seed(seed) with the "
            "SO(3) SDE on the GPU, stored in the *_initial_noise.pt file and "
            "returned by a wrapper around each SDE's prior_sampling in every "
            "arm (the original sampler still runs first; each arm logs "
            "whether its own seeded draw equals the stored one). Per-step "
            "noise: zero in every arm, because noise=0.0 gives t_hat == t; "
            "upstream still draws z in the pre-step Euler-Maruyama update "
            "and multiplies it by sqrt(|t_hat - t|) = 0, the fork skips the "
            "draw; pre_step_noise in each arm records that the update was an "
            "exact no-op. Train-mode dropout masks come from the CUDA "
            "generator after the seeded prior draw, as in unmodified upstream."
        ),
        "metric_definitions": _METRIC_DEFINITIONS,
        "kabsch_self_test_rmsd": _self_test_kabsch(),
        "trees": {
            tree: {
                key: meta.get(key)
                for key in (
                    "tree_root",
                    "git",
                    "code_fingerprint",
                    "bioemu_file",
                    "bioemu_version",
                    "harness_file",
                )
            }
            for tree, meta in metas.items()
        },
        "versions": {
            tree: meta.get("versions") for tree, meta in metas.items()
        },
        "gpu": {tree: meta.get("gpu") for tree, meta in metas.items()},
        "numerics": {
            tree: meta.get("numerics") for tree, meta in metas.items()
        },
        "env": {tree: meta.get("env") for tree, meta in metas.items()},
        "inputs": {
            key: any_meta.get(key)
            for key in (
                "checkpoint",
                "model_config",
                "dropout_p_in_model",
                "fasta",
                "embeds",
                "so3_cache",
                "denoiser",
                "initial_noise",
            )
        },
        "arms": arms,
        "arm_physics": {
            key: _physics(record["pos"]) for key, record in records.items()
        },
        "table": [
            _table_row(key, metrics) for key, metrics in pairwise.items()
        ],
        "pairwise": pairwise,
        "gate_numbers": {
            "A_plain_path_wrong": {
                "rmsd_nm_mean_a_vs_c": rmsd("a_vs_c"),
                "rmsd_nm_mean_a_vs_d": rmsd("a_vs_d"),
                "rmsd_nm_mean_a_vs_d_fixed_r": rmsd("a_vs_d_fixed_r"),
                "affine_kernel_layout": {
                    key: kernel_stats(key)
                    for key in ("c", "c_eager", "d", "d_fixed_r")
                },
            },
            "C_eval_vs_train": {
                "rmsd_nm_mean_a_vs_b": rmsd("a_vs_b"),
                "a_repeat_max_abs_pos": arms.get("a", {}).get(
                    "repeat_max_abs_pos"
                ),
                "b_repeat_max_abs_pos": arms.get("b", {}).get(
                    "repeat_max_abs_pos"
                ),
                "dropout_p_in_model": any_meta.get("dropout_p_in_model"),
            },
        },
        "files": sorted(path.name for path in out.glob(f"{args.prefix}_*")),
    }
    path = out / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"WROTE {path}")
    for row in summary["table"]:
        print(json.dumps(row))


def main(argv: Sequence[str] | None = None) -> None:
    """Runs the `run` or `compare` command."""
    args = _parse_args(argv)
    if args.command == "run":
        _run(args)
    else:
        _compare(args)


if __name__ == "__main__":
    main()
