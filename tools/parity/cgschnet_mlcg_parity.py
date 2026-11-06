"""Compares FlashMD's CGSchNet port with the original mlcg classes on CPU.

Loads a published checkpoint twice in one process: with `mlcg` importable,
as the reference, and through `flashmd.models.legacy_pickle`, which maps every
pickled `mlcg` class to its FlashMD port. Both models then evaluate the same
perturbed configurations, and the script reports the largest energy and force
differences per prior term and in total.

CPU tensors take the PyTorch code path in both packages, so this checks the
port and the class remap, not the Triton kernels.

Usage:

    PYTHONPATH=/path/to/mlcg/src:/path/to/flashmd/src \
        python tools/parity/cgschnet_mlcg_parity.py \
        --checkpoint model_and_prior.pt \
        --configurations 1enh_configurations.pt \
        --replicas 8 --dtype float64 --output parity.json
"""

import argparse
from collections.abc import Callable, Mapping, Sequence
import json
import pathlib
import sys
from typing import Any

import torch
from torch_geometric.data import collate as pyg_collate

_TOLERANCE = {torch.float32: 1e-4, torch.float64: 1e-9}


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare FlashMD CGSchNet with mlcg on CPU."
    )
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--configurations", type=pathlib.Path, required=True)
    parser.add_argument("--replicas", type=int, default=8)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dtype", choices=("float32", "float64"), default="float64"
    )
    parser.add_argument("--output", type=pathlib.Path)
    return parser.parse_args(argv)


def _batch(
    configurations: Sequence[Any],
    data_cls: type,
    replicas: int,
    noise: torch.Tensor,
    dtype: torch.dtype,
) -> Any:
    """Collates `replicas` copies of the configurations with fixed noise."""
    items = [
        configurations[i % len(configurations)].clone() for i in range(replicas)
    ]
    batch, _, _ = pyg_collate.collate(
        data_cls, items, increment=True, add_batch=True
    )
    batch.pos = (batch.pos.to(torch.float64) + noise).to(dtype)
    batch.pos.requires_grad_(True)
    return batch


def _flatten(out: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flattens the nested model output into `term.quantity` keys."""
    flat: dict[str, Any] = {}
    for key, value in out.items():
        if isinstance(value, Mapping):
            flat.update(_flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value.detach().to(torch.float64)
    return flat


def _evaluate(
    load_model: Callable[[pathlib.Path], torch.nn.Module],
    load_data: Callable[[pathlib.Path], Sequence[Any]],
    data_cls: type,
    args: argparse.Namespace,
    noise: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Loads one model, evaluates it, and returns flattened outputs."""
    dtype = getattr(torch, args.dtype)
    model = load_model(args.checkpoint).eval().to(dtype)
    configurations = load_data(args.configurations)
    batch = _batch(configurations, data_cls, args.replicas, noise, dtype)
    return _flatten(model(batch).out)


def _reference_outputs(
    args: argparse.Namespace, noise: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Evaluates the checkpoint with the original mlcg classes."""
    from mlcg.data import AtomicData
    from mlcg.nn import pyg_forward_compatibility

    def load_model(path: pathlib.Path) -> torch.nn.Module:
        model = pyg_forward_compatibility.load_and_adapt_old_checkpoint(path)
        # The FlashSchNet fork of mlcg reads flags that old checkpoints lack;
        # its simulation loader sets them to False the same way. CPU tensors
        # ignore them, and upstream mlcg never reads them.
        for module in model.modules():
            for flag in ("use_triton", "use_fused_tanh_linear"):
                name = type(module).__name__
                if name in ("CFConv", "InteractionBlock") and not hasattr(
                    module, flag
                ):
                    setattr(module, flag, False)
        return model

    return _evaluate(
        load_model,
        lambda path: torch.load(path, weights_only=False),
        AtomicData,
        args,
        noise,
    )


def _flashmd_outputs(
    args: argparse.Namespace, noise: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Evaluates the checkpoint through FlashMD's class remap."""
    from flashmd.data import AtomicData
    from flashmd.models import legacy_pickle
    from flashmd.models import pyg_forward_compatibility

    model = pyg_forward_compatibility.load_and_adapt_old_checkpoint(
        args.checkpoint
    )
    leftovers = {
        type(m).__module__
        for m in model.modules()
        if type(m).__module__.startswith("mlcg")
    }
    if leftovers:
        raise RuntimeError(f"mlcg classes survived the remap: {leftovers}")
    return _evaluate(
        lambda _: model,
        legacy_pickle.load,
        AtomicData,
        args,
        noise,
    )


def _compare(
    reference: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, float]]:
    """Returns max absolute and relative differences per output key."""
    if reference.keys() != candidate.keys():
        missing = sorted(set(reference) ^ set(candidate))
        raise RuntimeError(f"Output keys differ: {missing}")
    report = {}
    for key, ref in reference.items():
        diff = (candidate[key] - ref).abs().max().item()
        scale = ref.abs().max().item()
        report[key] = {
            "max_abs": diff,
            "max_rel": diff / scale if scale > 0 else diff,
        }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the comparison and returns a process exit code."""
    args = _parse_args(argv)
    torch.manual_seed(args.seed)
    probe = torch.load(args.configurations, weights_only=False)
    num_atoms = sum(
        probe[i % len(probe)].pos.shape[0] for i in range(args.replicas)
    )
    noise = args.noise * torch.randn(num_atoms, 3, dtype=torch.float64)
    del probe

    reference = _reference_outputs(args, noise)
    candidate = _flashmd_outputs(args, noise)
    report = _compare(reference, candidate)

    tolerance = _TOLERANCE[getattr(torch, args.dtype)]
    worst_key = max(report, key=lambda k: report[k]["max_rel"])
    worst = report[worst_key]["max_rel"]
    summary = {
        "dtype": args.dtype,
        "replicas": args.replicas,
        "num_atoms": num_atoms,
        "tolerance_rel": tolerance,
        "worst_key": worst_key,
        "worst_rel": worst,
        "total": {k: report[k] for k in ("energy", "forces")},
        "terms": report,
        "passed": worst <= tolerance,
    }
    if args.output is not None:
        args.output.write_text(json.dumps(summary, indent=2))
    print(
        json.dumps({k: v for k, v in summary.items() if k != "terms"}, indent=2)
    )
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
