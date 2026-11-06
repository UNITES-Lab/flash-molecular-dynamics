"""GPU correctness gates and golden outputs for CGSchNet.

FlashSchNet reads its kernel switches (`MLCG_USE_*`) once at import, so each
arm runs in its own process:

    python tools/gates/cgschnet_gpu_gates.py run --arm flash --model public \
        --checkpoint model_and_prior.pt \
        --configurations 1enh_configurations.pt --out results/
    python tools/gates/cgschnet_gpu_gates.py compare --out results/

Arms:
    reference_fp64: every switch off, float64. Ground truth.
    reference: every switch off, float32.
    reference_detached_envelope: as `reference`, but the CFConv cutoff
        envelope is detached from the graph. If `flash` matches this arm and
        not `reference`, the fused CFConv drops the envelope derivative.
    flash: every switch on, float32, no W16A16.
    flash_w16a16: `flash` plus the W16A16 filter networks the simulation
        applies by default.

Models:
    public: the published checkpoint with its priors.
    gauss128: random SchNet with `GaussianBasis`, which enables the fused
        distance and RBF kernel.
    exp256: random SchNet with 256 filters, beyond the 128-wide block of the
        source-CSR backward kernel.
"""

import argparse
from collections.abc import Mapping, Sequence
import json
import os
import pathlib
import sys
import time
import traceback
from typing import Any

_SWITCHES = (
    "MLCG_USE_TRITON_MESSAGE_PASSING",
    "MLCG_USE_FUSED_RBF",
    "MLCG_USE_FUSED_TANH_LINEAR",
    "MLCG_USE_CSR",
    "MLCG_USE_SRC_CSR_GRAD_X",
)
_ARMS = (
    "reference_fp64",
    "reference",
    "reference_detached_envelope",
    "flash",
    "flash_w16a16",
)
_MODELS = ("public", "gauss128", "exp256")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="CGSchNet GPU gates.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument(
        "--arm",
        required=True,
        help=(
            f"One of {', '.join(_ARMS)}, or only:<SWITCH> / without:<SWITCH> "
            "to isolate one kernel switch in float32."
        ),
    )
    run.add_argument("--model", choices=_MODELS, required=True)
    run.add_argument("--checkpoint", type=pathlib.Path, required=True)
    run.add_argument("--configurations", type=pathlib.Path, required=True)
    run.add_argument("--replicas", type=int, default=8)
    run.add_argument("--noise", type=float, default=0.05)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--out", type=pathlib.Path, required=True)
    run.add_argument("--device", default="cuda")
    compare = sub.add_parser("compare")
    compare.add_argument("--out", type=pathlib.Path, required=True)
    return parser.parse_args(argv)


def _set_switches(arm: str) -> None:
    """Sets the import-time kernel switches for an arm.

    Raises:
        ValueError: If the arm or the switch it names is unknown.
    """
    if arm.startswith(("only:", "without:")):
        mode, switch = arm.split(":", 1)
        if switch not in _SWITCHES:
            raise ValueError(f"Unknown switch {switch!r}.")
        for name in _SWITCHES:
            on = (name == switch) if mode == "only" else (name != switch)
            os.environ[name] = "1" if on else "0"
        return
    if arm not in _ARMS:
        raise ValueError(f"Unknown arm {arm!r}.")
    value = "1" if arm.startswith("flash") else "0"
    for name in _SWITCHES:
        os.environ[name] = value


def _synthetic_model(kind: str) -> Any:
    """Builds a randomly initialized SchNet for the latent-bug gates."""
    import torch

    from flashmd.models import cutoff as cutoff_lib
    from flashmd.models import gradients
    from flashmd.models import radial_basis
    from flashmd.models import schnet

    torch.manual_seed(1234)
    cutoff = cutoff_lib.CosineCutoff(0.0, 15.0)
    if kind == "gauss128":
        rbf = radial_basis.GaussianBasis(cutoff, num_rbf=64)
        num_filters = 128
    else:
        rbf = radial_basis.ExpNormalBasis(cutoff, num_rbf=64)
        num_filters = 256
    model = schnet.StandardSchNet(
        rbf,
        cutoff,
        output_hidden_layer_widths=[128, 64],
        hidden_channels=128,
        embedding_size=25,
        num_filters=num_filters,
        num_interactions=2,
    )
    return gradients.GradientsOut(model, targets="forces")


class _DetachedEnvelope:
    """Factory for a cutoff wrapper whose output carries no gradient."""

    @staticmethod
    def wrap(model: Any) -> int:
        """Detaches every CFConv cutoff envelope in place.

        Args:
            model: Module tree containing CFConv layers.

        Returns:
            Number of CFConv layers changed.
        """
        import torch

        class Detached(torch.nn.Module):
            def __init__(self, inner: torch.nn.Module) -> None:
                super().__init__()
                self.inner = inner
                self.cutoff_lower = inner.cutoff_lower
                self.cutoff_upper = inner.cutoff_upper

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.inner(x).detach()

        count = 0
        for module in model.modules():
            if type(module).__name__ == "CFConv":
                module.cutoff = Detached(module.cutoff)
                count += 1
        return count


def _module_flags(model: Any) -> dict[str, Any]:
    """Collects the per-module kernel flags actually set on the model."""
    flags: dict[str, Any] = {}
    for module in model.modules():
        name = type(module).__name__
        if name == "CFConv":
            flags.setdefault("CFConv.use_triton", set()).add(
                bool(getattr(module, "use_triton", False))
            )
        elif name == "InteractionBlock":
            flags.setdefault(
                "InteractionBlock.use_fused_tanh_linear", set()
            ).add(bool(getattr(module, "use_fused_tanh_linear", False)))
        elif name in ("GaussianBasis", "ExpNormalBasis"):
            flags["rbf"] = name
    return {k: sorted(v) if isinstance(v, set) else v for k, v in flags.items()}


def _run(args: argparse.Namespace) -> int:
    """Evaluates one arm and saves energies and forces."""
    _set_switches(args.arm)
    import torch
    from torch_geometric.data import collate as pyg_collate

    from flashmd.data import AtomicData
    from flashmd.models import legacy_pickle
    from flashmd.models import pyg_forward_compatibility

    dtype = torch.float64 if args.arm == "reference_fp64" else torch.float32
    device = torch.device(args.device)
    on_cuda = device.type == "cuda"
    record: dict[str, Any] = {
        "arm": args.arm,
        "model": args.model,
        "switches": {k: os.environ[k] for k in _SWITCHES},
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if on_cuda else "cpu",
        "dtype": str(dtype),
    }
    stem = args.out / f"{args.model}__{args.arm.replace(':', '-')}"
    try:
        if args.model == "public":
            model = pyg_forward_compatibility.load_and_adapt_old_checkpoint(
                args.checkpoint
            )
        else:
            model = _synthetic_model(args.model)
        model = model.eval().to(device=device, dtype=dtype)
        if args.arm == "reference_detached_envelope":
            record["detached_cfconv"] = _DetachedEnvelope.wrap(model)
        if args.arm == "flash_w16a16":
            from flashmd.models import gptq

            model = gptq.apply_gptq_w16a16_to_model(model)
            gptq.validate_gptq_w16a16(model)
        for param in model.parameters():
            param.requires_grad_(False)
        record["module_flags"] = _module_flags(model)

        configurations = legacy_pickle.load(args.configurations)
        items = [
            configurations[i % len(configurations)].clone()
            for i in range(args.replicas)
        ]
        batch, _, _ = pyg_collate.collate(
            AtomicData, items, increment=True, add_batch=True
        )
        generator = torch.Generator().manual_seed(args.seed)
        noise = args.noise * torch.randn(
            batch.pos.shape, dtype=torch.float64, generator=generator
        )
        batch.pos = (batch.pos.to(torch.float64) + noise).to(dtype)
        batch = batch.to(device)

        outputs = []
        timings = []
        for _ in range(3):
            data = batch.clone()
            data.pos.requires_grad_(True)
            if on_cuda:
                torch.cuda.synchronize()
            start = time.perf_counter()
            out = model(data).out
            if on_cuda:
                torch.cuda.synchronize()
            timings.append(time.perf_counter() - start)
            outputs.append(out)
        out = outputs[-1]
        term = out.get("SchNet", out)
        result = {
            "energy": out.get("energy", term["energy"]).detach().cpu(),
            "forces": out.get("forces", term["forces"]).detach().cpu(),
            "schnet_energy": term["energy"].detach().cpu(),
            "schnet_forces": term["forces"].detach().cpu(),
        }
        repeat = outputs[0]
        repeat_term = repeat.get("SchNet", repeat)
        record["run_to_run_max_abs_schnet_forces"] = (
            (repeat_term["forces"] - term["forces"]).abs().max().item()
        )
        record["seconds_per_eval"] = timings[1:]
        record["num_atoms"] = int(batch.pos.shape[0])
        record["status"] = "ok"
        torch.save(result, stem.with_suffix(".pt"))
    except Exception as error:
        record["status"] = "error"
        record["error"] = f"{type(error).__name__}: {error}"
        record["traceback"] = traceback.format_exc()
    stem.with_suffix(".json").write_text(json.dumps(record, indent=2))
    print(json.dumps({k: v for k, v in record.items() if k != "traceback"}))
    return 0


def _diff(a: Mapping[str, Any], b: Mapping[str, Any], key: str) -> dict:
    """Returns max absolute and relative differences of one quantity."""
    import torch

    x = a[key].to(torch.float64)
    y = b[key].to(torch.float64)
    diff = (x - y).abs().max().item()
    scale = y.abs().max().item()
    return {"max_abs": diff, "max_rel": diff / scale if scale else diff}


def _compare(args: argparse.Namespace) -> int:
    """Builds the pairwise difference table for every model."""
    import torch

    pairs = (
        ("reference", "reference_fp64"),
        ("flash", "reference_fp64"),
        ("flash", "reference"),
        ("flash", "reference_detached_envelope"),
        ("reference_detached_envelope", "reference"),
        ("flash_w16a16", "reference_fp64"),
        ("flash_w16a16", "flash"),
    )
    table: dict[str, Any] = {}
    for model in _MODELS:
        loaded: dict[str, Any] = {}
        status: dict[str, str] = {}
        for arm in _ARMS:
            meta = args.out / f"{model}__{arm}.json"
            if not meta.exists():
                continue
            record = json.loads(meta.read_text())
            status[arm] = record["status"] + (
                f" ({record['error']})" if record["status"] != "ok" else ""
            )
            tensor_path = args.out / f"{model}__{arm}.pt"
            if record["status"] == "ok" and tensor_path.exists():
                loaded[arm] = torch.load(tensor_path)
        rows = {}
        isolation = sorted(
            path.stem.split("__", 1)[1]
            for path in args.out.glob(f"{model}__only-*.pt")
        ) + sorted(
            path.stem.split("__", 1)[1]
            for path in args.out.glob(f"{model}__without-*.pt")
        )
        for arm in isolation:
            record = json.loads((args.out / f"{model}__{arm}.json").read_text())
            status[arm] = record["status"]
            loaded[arm] = torch.load(args.out / f"{model}__{arm}.pt")
        all_pairs = list(pairs) + [(arm, "reference") for arm in isolation]
        for left, right in all_pairs:
            if left in loaded and right in loaded:
                rows[f"{left} vs {right}"] = {
                    key: _diff(loaded[left], loaded[right], key)
                    for key in ("schnet_forces", "forces", "energy")
                }
        table[model] = {"status": status, "pairs": rows}
    (args.out / "summary.json").write_text(json.dumps(table, indent=2))
    for model, entry in table.items():
        print(f"\n## {model}")
        for arm, state in entry["status"].items():
            print(f"- {arm}: {state}")
        print("| pair | SchNet forces rel | total forces rel | energy rel |")
        print("|---|---|---|---|")
        for pair, row in entry["pairs"].items():
            print(
                f"| {pair} | {row['schnet_forces']['max_rel']:.2e} | "
                f"{row['forces']['max_rel']:.2e} | "
                f"{row['energy']['max_rel']:.2e} |"
            )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatches to `run` or `compare`."""
    args = _parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    return _run(args) if args.command == "run" else _compare(args)


if __name__ == "__main__":
    sys.exit(main())
