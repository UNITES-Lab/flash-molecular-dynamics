#!/usr/bin/env python
"""Benchmark dense 1ENH replica scaling with baseline and full FlashMD."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
SCHNET = ROOT / "flash-schnet"
CONFIG = (
    SCHNET
    / "data/simulating_a_trained_cg_model/"
    "1enh_sim_langevin_epoch_73-dev-bsz64-300k.yaml"
)
PYTHON = Path("/home/pingzhi/anaconda3/envs/mlcg/bin/python")
RUNNER = SCHNET / "benchmark_runner.py"
OUTPUT = ROOT / "submission/data/cgschnet_1enh_full_scaling"
CSV_PATH = OUTPUT / "cgschnet_1enh_full_scaling.csv"
BATCH_SIZES = (
    1,
    2,
    4,
    8,
    12,
    16,
    24,
    32,
    48,
    64,
    96,
    128,
    192,
    256,
    384,
    512,
    640,
    768,
)
SOURCE_CHECKPOINT = (
    SCHNET
    / "data/simulating_a_trained_cg_model/outputs/"
    "16ns-full-sim-1enh_20260129_192001_checkpoint_0250.pt"
)
SOURCE_BATCH_SIZE = 64
N_BEADS = 269
CHECKPOINT_STEP = 251000
WARMUP_STEPS = 25000
MEASUREMENT_STEPS = 25000
PROTOCOL_STEPS = WARMUP_STEPS + MEASUREMENT_STEPS
GROUND_TRUTH = {
    "baseline": {
        16: 453.6154161001888,
        32: 513.8979995325768,
        64: 477.0,
    },
    "flash": {
        16: 964.9737179412782,
        32: 1938.9574378284544,
        64: 3095.0,
    },
}
GROUND_TRUTH_TOLERANCE = 0.15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--series",
        choices=("baseline", "flash", "both"),
        default="both",
    )
    parser.add_argument(
        "--batches",
        type=int,
        nargs="+",
        choices=BATCH_SIZES,
        default=list(BATCH_SIZES),
    )
    parser.add_argument("--n-timesteps", type=int, default=PROTOCOL_STEPS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--use-prepared-inputs", action="store_true")
    parser.add_argument("--no-collect", action="store_true")
    return parser.parse_args()


def result_paths(series: str, batch_size: int) -> tuple[Path, Path, Path]:
    directory = OUTPUT / series / f"bs{batch_size}"
    return (
        directory / "run_metrics.json",
        directory / "run.log",
        directory / "failure.json",
    )


def checkpoint_path(batch_size: int) -> Path:
    return OUTPUT / "inputs" / f"checkpoint_step{CHECKPOINT_STEP}_bs{batch_size}.pt"


def config_path(batch_size: int) -> Path:
    return OUTPUT / "inputs" / f"benchmark_step{CHECKPOINT_STEP}_bs{batch_size}.yaml"


def prepare_input(batch_size: int, n_timesteps: int) -> Path:
    target_checkpoint = checkpoint_path(batch_size)
    target_config = config_path(batch_size)
    target_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    source = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    source_pos = source["pos"].reshape(SOURCE_BATCH_SIZE, N_BEADS, 3)
    source_velocities = source["velocities"].reshape(
        SOURCE_BATCH_SIZE, N_BEADS, 3
    )
    indices = torch.arange(batch_size) % SOURCE_BATCH_SIZE
    checkpoint = {
        "pos": source_pos[indices].reshape(batch_size * N_BEADS, 3).contiguous(),
        "velocities": source_velocities[indices]
        .reshape(batch_size * N_BEADS, 3)
        .contiguous(),
        "current_timestep": 0,
        "export_interval": n_timesteps,
        "save_interval": n_timesteps,
        "log_interval": n_timesteps,
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "source_physical_step": CHECKPOINT_STEP,
    }
    torch.save(checkpoint, target_checkpoint)

    config = yaml.safe_load(CONFIG.read_text())
    data_dir = CONFIG.parent
    config["model_file"] = str(data_dir / config["model_file"])
    config["structure_file"] = str(data_dir / config["structure_file"])
    config["batch_size"] = batch_size
    simulation = config["simulation"]
    simulation.update(
        {
            "n_timesteps": n_timesteps,
            "save_forces": False,
            "save_energies": False,
            "save_interval": n_timesteps,
            "export_interval": n_timesteps,
            "log_interval": n_timesteps,
            "create_checkpoints": False,
            "read_checkpoint_file": str(target_checkpoint),
            "filename": f"checkpoint_scaling_bs{batch_size}",
            "add_timestamp": False,
            "random_seed": 233,
        }
    )
    target_config.write_text(yaml.safe_dump(config, sort_keys=False))
    return target_config


def run_one(
    *,
    series: str,
    batch_size: int,
    n_timesteps: int,
    force: bool,
    use_prepared_inputs: bool,
) -> None:
    metrics_path, log_path, failure_path = result_paths(series, batch_size)
    if not force and (metrics_path.exists() or failure_path.exists()):
        print(f"SKIP {series} bs={batch_size}", flush=True)
        return

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.unlink(missing_ok=True)
    failure_path.unlink(missing_ok=True)
    prefix = metrics_path.with_name("run")
    prepared_config = config_path(batch_size)
    if use_prepared_inputs:
        if not prepared_config.exists() or not checkpoint_path(batch_size).exists():
            raise FileNotFoundError(
                f"Prepared input missing for batch size {batch_size}"
            )
    else:
        prepared_config = prepare_input(batch_size, n_timesteps)

    enabled = "1" if series == "flash" else "0"
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONPATH": os.pathsep.join(
                [str(SCHNET / "src"), env.get("PYTHONPATH", "")]
            ),
            "PYTHONUNBUFFERED": "1",
            "MLCG_USE_TRITON_MESSAGE_PASSING": enabled,
            "MLCG_USE_FUSED_RBF": enabled,
            "MLCG_USE_CSR": enabled,
            "MLCG_USE_SRC_CSR_GRAD_X": enabled,
            "MLCG_BENCHMARK_SKIP_OUTPUT_IO": "1",
            "MLCG_BENCHMARK_SECOND_HALF": "1",
        }
    )
    compile_model = series == "flash" and batch_size > 1
    command = [
        str(PYTHON),
        str(RUNNER),
        "--config",
        str(prepared_config),
        "--batch_size",
        str(batch_size),
        "--n_timesteps",
        str(n_timesteps),
        "--log_interval",
        str(max(1, n_timesteps // 4)),
        "--output_prefix",
        str(prefix),
        "--compile_model",
        "true" if compile_model else "false",
    ]
    if series == "flash":
        command.extend(["--gptq", "w16a16"])

    print(f"RUN  {series} bs={batch_size}", flush=True)
    with log_path.open("w") as log:
        completed = subprocess.run(
            command,
            cwd=SCHNET,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode == 0:
        print(f"DONE {series} bs={batch_size}", flush=True)
        return

    log_text = log_path.read_text(errors="ignore")
    lowered = log_text.lower()
    status = (
        "oom"
        if "out of memory" in lowered or "cuda_error_out_of_memory" in lowered
        else "failed"
    )
    failure_path.write_text(
        json.dumps(
            {
                "series": series,
                "batch_size": batch_size,
                "status": status,
                "returncode": completed.returncode,
                "command": command,
                "log": str(log_path.relative_to(ROOT)),
                "log_tail": "\n".join(log_text.splitlines()[-80:]),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"{status.upper()} {series} bs={batch_size}", flush=True)


def collect() -> None:
    fields = (
        "series",
        "batch_size",
        "status",
        "throughput",
        "ms_per_timestep",
        "peak_memory_allocated_gb",
        "peak_memory_reserved_gb",
        "n_timesteps",
        "compile_model",
        "gptq",
        "checkpoint_step",
        "warmup_steps",
        "measurement_steps",
        "ground_truth",
        "relative_error",
        "within_ground_truth_tolerance",
        "source",
    )
    rows: list[dict[str, object]] = []
    for series in ("baseline", "flash"):
        for batch_size in BATCH_SIZES:
            metrics_path, _log_path, failure_path = result_paths(series, batch_size)
            if metrics_path.exists():
                data = json.loads(metrics_path.read_text())
                steady = data["steady_state_metrics"]
                config = data["config"]
                ground_truth = GROUND_TRUTH.get(series, {}).get(batch_size)
                relative_error = (
                    abs(float(steady["throughput"]) - ground_truth) / ground_truth
                    if ground_truth is not None
                    else ""
                )
                rows.append(
                    {
                        "series": series,
                        "batch_size": batch_size,
                        "status": "ok",
                        "throughput": steady["throughput"],
                        "ms_per_timestep": steady["ms_per_timestep"],
                        "peak_memory_allocated_gb": steady[
                            "peak_memory_allocated_gb"
                        ],
                        "peak_memory_reserved_gb": steady[
                            "peak_memory_reserved_gb"
                        ],
                        "n_timesteps": config["n_timesteps"],
                        "compile_model": config["compile_model"],
                        "gptq": config["gptq"] or "",
                        "checkpoint_step": CHECKPOINT_STEP,
                        "warmup_steps": config["n_timesteps"] // 2,
                        "measurement_steps": config["n_timesteps"]
                        - config["n_timesteps"] // 2,
                        "ground_truth": ground_truth or "",
                        "relative_error": relative_error,
                        "within_ground_truth_tolerance": (
                            relative_error <= GROUND_TRUTH_TOLERANCE
                            if relative_error != ""
                            else ""
                        ),
                        "source": str(metrics_path.relative_to(ROOT)),
                    }
                )
            elif failure_path.exists():
                failure = json.loads(failure_path.read_text())
                rows.append(
                    {
                        "series": series,
                        "batch_size": batch_size,
                        "status": failure["status"],
                        "throughput": "",
                        "ms_per_timestep": "",
                        "peak_memory_allocated_gb": "",
                        "peak_memory_reserved_gb": "",
                        "n_timesteps": "",
                        "compile_model": series == "flash",
                        "gptq": "w16a16" if series == "flash" else "",
                        "checkpoint_step": CHECKPOINT_STEP,
                        "warmup_steps": "",
                        "measurement_steps": "",
                        "ground_truth": GROUND_TRUTH.get(series, {}).get(
                            batch_size, ""
                        ),
                        "relative_error": "",
                        "within_ground_truth_tolerance": "",
                        "source": str(failure_path.relative_to(ROOT)),
                    }
                )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(CSV_PATH, flush=True)


def main() -> None:
    args = parse_args()
    if args.prepare_only:
        for batch_size in args.batches:
            prepared = prepare_input(batch_size, args.n_timesteps)
            print(f"PREP {prepared}", flush=True)
        return

    series_values = (
        ("baseline", "flash") if args.series == "both" else (args.series,)
    )
    for series in series_values:
        for batch_size in args.batches:
            run_one(
                series=series,
                batch_size=batch_size,
                n_timesteps=args.n_timesteps,
                force=args.force,
                use_prepared_inputs=args.use_prepared_inputs,
            )
            if not args.no_collect:
                collect()
    if not args.no_collect:
        collect()


if __name__ == "__main__":
    main()
