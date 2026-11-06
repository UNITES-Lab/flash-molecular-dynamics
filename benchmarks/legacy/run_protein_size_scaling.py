#!/usr/bin/env python
"""Run and collect fixed-batch protein-size scaling benchmarks for FlashMD."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "submission/data/protein_size_scaling"
MANIFEST = DATA / "protein_manifest.json"
RESULTS = DATA / "results"

SCHNET = ROOT / "flash-schnet"
MACE = ROOT / "flash-mace"
BIOEMU = ROOT / "flash-bioemu"
BIOEMU_OFFICIAL = ROOT / "bioemu-official-benchmark"

MLCG_PYTHON = Path("/home/pingzhi/anaconda3/envs/mlcg/bin/python")
MACE_PYTHON = Path("/playpen-shared/pingzhi/anaconda3/envs/flash-mace/bin/python")
BIOEMU_PYTHON = Path("/home/pingzhi/anaconda3/envs/bioemu/bin/python")
BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        choices=["schnet", "mace", "bioemu", "all", "collect"],
        help="Benchmark family to run, all families, or collection only.",
    )
    parser.add_argument("--force", action="store_true", help="Rerun completed jobs.")
    parser.add_argument(
        "--batch-size",
        type=int,
        choices=BATCH_SIZES,
        default=1,
        help="Number of independent replicas/samples per protein size.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional system IDs to run (primarily useful for smoke tests).",
    )
    return parser.parse_args()


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def selected(items: Iterable[dict], only: list[str] | None) -> list[dict]:
    values = list(items)
    if only is None:
        return values
    requested = set(only)
    return [item for item in values if str(item["id"]) in requested]


def run_logged(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    failure_path: Path,
) -> bool:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"RUN {log_path.relative_to(ROOT)}", flush=True)
    if failure_path.exists():
        failure_path.unlink()
    try:
        with log_path.open("w") as log:
            subprocess.run(
                command,
                cwd=cwd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
    except subprocess.CalledProcessError as error:
        log_text = log_path.read_text(errors="ignore")
        lowered = log_text.lower()
        kind = (
            "oom"
            if "out of memory" in lowered
            or "cuda error: out of memory" in lowered
            or "cuda_error_out_of_memory" in lowered
            else "failed"
        )
        failure = {
            "status": kind,
            "returncode": error.returncode,
            "command": command,
            "log": str(log_path.relative_to(ROOT)),
            "log_tail": "\n".join(log_text.splitlines()[-40:]),
        }
        failure_path.write_text(json.dumps(failure, indent=2) + "\n")
        print(f"{kind.upper()} {failure_path.relative_to(ROOT)}", flush=True)
        return False
    return True


def results_root(batch_size: int) -> Path:
    return RESULTS if batch_size == 1 else DATA / f"results_bs{batch_size}"


def run_schnet(
    manifest: dict,
    *,
    force: bool,
    only: list[str] | None,
    batch_size: int,
) -> None:
    systems = selected(
        manifest["schnet_native"] + manifest.get("schnet_shared_from_mace", []),
        only,
    )
    common_env = os.environ.copy()
    common_env["CUDA_VISIBLE_DEVICES"] = "0"
    common_env["PYTHONPATH"] = os.pathsep.join(
        [str(SCHNET / "src"), common_env.get("PYTHONPATH", "")]
    )
    for system in systems:
        for series in ("baseline", "flash"):
            out_dir = results_root(batch_size) / "schnet" / series / str(system["id"])
            prefix = out_dir / "run"
            metrics = Path(f"{prefix}_metrics.json")
            failure = out_dir / "failure.json"
            if (metrics.exists() or failure.exists()) and not force:
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            env = common_env.copy()
            enabled = "1" if series == "flash" else "0"
            env.update(
                {
                    "MLCG_USE_TRITON_MESSAGE_PASSING": enabled,
                    "MLCG_USE_FUSED_RBF": enabled,
                    "MLCG_USE_CSR": enabled,
                    "MLCG_USE_SRC_CSR_GRAD_X": enabled,
                    "MLCG_BENCHMARK_SKIP_OUTPUT_IO": "1",
                    "MLCG_BENCHMARK_SECOND_HALF": "1",
                }
            )
            command = [
                str(MLCG_PYTHON),
                str(SCHNET / "benchmark_runner.py"),
                "--config",
                str(ROOT / system["config"]),
                "--batch_size",
                str(batch_size),
                "--n_timesteps",
                "1000",
                "--log_interval",
                "100",
                "--output_prefix",
                str(prefix),
                "--compile_model",
                "false",
            ]
            if series == "flash":
                command.extend(["--gptq", "w16a16"])
            run_logged(
                command,
                cwd=SCHNET,
                env=env,
                log_path=out_dir / "run.log",
                failure_path=failure,
            )


def run_mace(
    manifest: dict,
    *,
    force: bool,
    only: list[str] | None,
    batch_size: int,
) -> None:
    systems = selected(
        manifest["mace_native"] + manifest["mace_shared_from_schnet"], only
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(MACE), env.get("PYTHONPATH", "")]
    )
    for system in systems:
        for series in ("baseline", "flash"):
            out_dir = results_root(batch_size) / "mace" / series
            output = out_dir / f"{system['id']}.json"
            failure = out_dir / f"{system['id']}.failure.json"
            if (output.exists() or failure.exists()) and not force:
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            backend = "cueq_f64" if series == "baseline" else "cueq_triton_fp32"
            command = [
                str(MACE_PYTHON),
                str(MACE / "scripts/benchmark_mace_protein_md.py"),
                "--pdb",
                str(ROOT / system["pdb"]),
                "--model-size",
                "medium",
                "--batch-size",
                str(batch_size),
                "--steps",
                "12",
                "--measure-window",
                "8",
                "--mode",
                "single",
                "--backend",
                backend,
                "--log-every",
                "1000",
                "--output",
                str(output),
            ]
            if series == "flash":
                command.extend(
                    [
                        "--allow-tf32",
                        "--matmul-precision",
                        "high",
                        "--compile-core-modules",
                        "--fixed-onehot-fctp",
                        "--energy-force-only",
                    ]
                )
            run_logged(
                command,
                cwd=MACE,
                env=env,
                log_path=out_dir / f"{system['id']}.log",
                failure_path=failure,
            )


def run_bioemu(
    manifest: dict,
    *,
    force: bool,
    only: list[str] | None,
    batch_size: int,
) -> None:
    contexts = selected(manifest["bioemu_contexts"], only)
    cache = ROOT / manifest["bioemu_embed_cache"]
    for context in contexts:
        for series in ("baseline", "flash"):
            repo = BIOEMU_OFFICIAL if series == "baseline" else BIOEMU
            script = repo / "dev/benchmark_bioemu_throughput.py"
            out_dir = results_root(batch_size) / "bioemu" / series / str(context["id"])
            summary = out_dir / "summary.json"
            failure = out_dir / "failure.json"
            if (summary.exists() or failure.exists()) and not force:
                continue
            env = os.environ.copy()
            env.update(
                {
                    "CUDA_VISIBLE_DEVICES": "0",
                    "JAX_PLATFORM_NAME": "cpu",
                    "PYTHONPATH": os.pathsep.join(
                        [str(repo / "src"), env.get("PYTHONPATH", "")]
                    ),
                }
            )
            command = [
                str(BIOEMU_PYTHON),
                str(script),
                "--sequence",
                str(ROOT / context["fasta"]),
                "--cache-embeds-dir",
                str(cache),
                "--cache-so3-dir",
                str(BIOEMU / "profiles/so3_cache"),
                "--batch-size",
                str(batch_size),
                "--steps",
                "9",
                "--warmup-runs",
                "2" if series == "baseline" else "5",
                "--timed-runs",
                "5",
                "--activation-steps",
                "none",
                "--output-dir",
                str(out_dir),
                "--jax-platform",
                "cpu",
            ]
            if series == "flash":
                command.extend(["--static-context-cache", "--cuda-graph-dpm"])
            run_logged(
                command,
                cwd=repo,
                env=env,
                log_path=out_dir / "run.log",
                failure_path=failure,
            )


def schnet_value(path: Path) -> tuple[float, float, float]:
    data = json.loads(path.read_text())
    steady_metrics = data.get("steady_state_metrics")
    if steady_metrics:
        return (
            float(steady_metrics["throughput"]),
            float(steady_metrics["peak_memory_allocated_gb"]),
            float(steady_metrics["peak_memory_reserved_gb"]),
        )
    windows = data.get("periodic_metrics", [])
    steady = windows[-5:] if len(windows) >= 5 else windows[1:]
    throughputs = [float(item["throughput"]) for item in steady]
    if not throughputs:
        throughputs = [float(data["final_metrics"]["throughput"])]
    return (
        statistics.median(throughputs),
        float(data["final_metrics"]["max_memory_allocated_gb"]),
        float(data["final_metrics"]["max_memory_reserved_gb"]),
    )


def mace_value(path: Path) -> tuple[float, float, float, int]:
    data = json.loads(path.read_text())
    latency = data["result"]["single"]["last_window_latency"]
    mean_ms = float(latency["mean_ms"])
    batch_size = int(data["config"]["batch_size"])
    result = data["result"]["single"]
    return (
        batch_size * 1000.0 / mean_ms,
        float(result["last_window_peak_allocated_mb"]) / 1000.0,
        float(result["last_window_peak_reserved_mb"]) / 1000.0,
        int(data["system"]["num_edges_per_replica"]),
    )


def bioemu_value(path: Path) -> tuple[float, float, float]:
    data = json.loads(path.read_text())
    cuda = data.get("cuda", {})
    return (
        float(data["throughput_samples_per_s_wall"]),
        float(cuda.get("peak_memory_allocated_bytes", 0)) / 1.0e9,
        float(cuda.get("peak_memory_reserved_bytes", 0)) / 1.0e9,
    )


def collect(manifest: dict, *, batch_size: int) -> Path:
    rows: list[dict[str, object]] = []
    root = results_root(batch_size)
    families = {
        "schnet": manifest["schnet_native"]
        + manifest.get("schnet_shared_from_mace", []),
        "mace": manifest["mace_native"] + manifest["mace_shared_from_schnet"],
        "bioemu": manifest["bioemu_contexts"],
    }
    for family, systems in families.items():
        for system in systems:
            for series in ("baseline", "flash"):
                if family == "schnet":
                    result_dir = root / family / series / str(system["id"])
                    path = result_dir / "run_metrics.json"
                    failure_path = result_dir / "failure.json"
                    if not path.exists():
                        if not failure_path.exists():
                            continue
                        failure = json.loads(failure_path.read_text())
                        throughput = allocated = reserved = ""
                        status = failure["status"]
                        error = failure.get("log_tail", "")
                        size = int(system["beads"])
                        unit = "steps/s"
                        edges = ""
                        path = failure_path
                    else:
                        throughput, allocated, reserved = schnet_value(path)
                        status = "ok"
                        error = ""
                        size = int(system["beads"])
                        unit = "steps/s"
                        edges = ""
                elif family == "mace":
                    result_dir = root / family / series
                    path = result_dir / f"{system['id']}.json"
                    failure_path = result_dir / f"{system['id']}.failure.json"
                    if not path.exists():
                        if not failure_path.exists():
                            continue
                        failure = json.loads(failure_path.read_text())
                        throughput = allocated = reserved = ""
                        status = failure["status"]
                        error = failure.get("log_tail", "")
                        edges = ""
                        path = failure_path
                    else:
                        throughput, allocated, reserved, edges = mace_value(path)
                        status = "ok"
                        error = ""
                    size = int(system["atoms"])
                    unit = "force evaluations/s"
                else:
                    result_dir = root / family / series / str(system["id"])
                    path = result_dir / "summary.json"
                    failure_path = result_dir / "failure.json"
                    if not path.exists():
                        if not failure_path.exists():
                            continue
                        failure = json.loads(failure_path.read_text())
                        throughput = allocated = reserved = ""
                        status = failure["status"]
                        error = failure.get("log_tail", "")
                        path = failure_path
                    else:
                        throughput, allocated, reserved = bioemu_value(path)
                        status = "ok"
                        error = ""
                    size = int(system["residues"])
                    unit = "9-step samples/s"
                    edges = size * size
                rows.append(
                    {
                        "family": family,
                        "batch_size": batch_size,
                        "series": series,
                        "system": system["id"],
                        "size": size,
                        "size_unit": {
                            "schnet": "beads",
                            "mace": "atoms",
                            "bioemu": "residues",
                        }[family],
                        "edges": edges,
                        "throughput": throughput,
                        "throughput_unit": unit,
                        "peak_allocated_gb": allocated,
                        "peak_reserved_gb": reserved,
                        "source": system.get("source", ",".join(system.get("sources", []))),
                        "embedding": system.get("embedding", ""),
                        "status": status,
                        "error": error,
                        "result_path": str(path.relative_to(ROOT)),
                    }
                )

    output = DATA / f"protein_size_scaling_bs{batch_size}.csv"
    fields = [
        "family",
        "batch_size",
        "series",
        "system",
        "size",
        "size_unit",
        "edges",
        "throughput",
        "throughput_unit",
        "peak_allocated_gb",
        "peak_reserved_gb",
        "source",
        "embedding",
        "status",
        "error",
        "result_path",
    ]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if batch_size == 1:
        shutil.copy2(output, DATA / "protein_size_scaling.csv")
    print(f"WROTE {output.relative_to(ROOT)} ({len(rows)} rows)", flush=True)
    return output


def main() -> None:
    args = parse_args()
    manifest = load_manifest()
    if args.target in {"schnet", "all"}:
        run_schnet(
            manifest,
            force=args.force,
            only=args.only,
            batch_size=args.batch_size,
        )
    if args.target in {"mace", "all"}:
        run_mace(
            manifest,
            force=args.force,
            only=args.only,
            batch_size=args.batch_size,
        )
    if args.target in {"bioemu", "all"}:
        run_bioemu(
            manifest,
            force=args.force,
            only=args.only,
            batch_size=args.batch_size,
        )
    collect(manifest, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
