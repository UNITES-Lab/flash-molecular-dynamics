#! /usr/bin/env python

import os
import sys

# Process --disable_optim BEFORE importing flashmd (env vars are read at import time)
_DISABLE_OPTIM = "--disable_optim" in sys.argv
if _DISABLE_OPTIM:
    sys.argv.remove("--disable_optim")
    for var in [
        "MLCG_USE_TRITON_MESSAGE_PASSING",
        "MLCG_USE_FUSED_RBF",
        "MLCG_USE_FUSED_TANH_LINEAR",
        "MLCG_USE_CSR",
        "MLCG_USE_SRC_CSR_GRAD_X",
    ]:
        os.environ[var] = "0"

import time
import torch
from torch.profiler import (
    profile,
    ProfilerActivity,
    schedule,
    tensorboard_trace_handler,
)

from flashmd.simulation import (
    parse_simulation_config,
    LangevinSimulation,
)
from flashmd.simulation.logging import logger, setup_simulation_logging


def main():
    torch.jit.set_fusion_strategy([("DYNAMIC", 3)])
    # to leverage the tensor core if available
    torch.set_float32_matmul_precision("high")

    # Build overrides if --disable_optim was passed
    disable_overrides = {}
    if _DISABLE_OPTIM:
        disable_overrides = {
            "compile_model": False,
            "gptq": None,
        }

    (
        model,
        initial_data_list,
        betas,
        simulation,
        profile_output,
    ) = parse_simulation_config(LangevinSimulation)

    # Apply --disable_optim overrides to simulation object
    if disable_overrides:
        for k, v in disable_overrides.items():
            setattr(simulation, k, v)
        logger.info("--disable_optim: compile_model=False, gptq=None")

    # Set up logging with file output based on simulation filename
    # Note: simulation.filename already includes timestamp if add_timestamp was True
    # So we use add_timestamp=False here to avoid double timestamps
    log_file = setup_simulation_logging(
        filename=simulation.filename,
        output_dir=".",  # simulation.filename already includes output_dir
        add_timestamp=False,
    )

    logger.info(f"Starting simulation with {LangevinSimulation.__name__}")
    logger.info(f"Simulation filename: {simulation.filename}")
    logger.info("=" * 50)
    logger.info("Simulation Parameters")
    logger.info("-" * 50)
    for key, value in sorted(vars(simulation).items()):
        if not key.startswith("_") and not callable(value):
            # Skip large objects like tensors
            if hasattr(value, "shape"):
                logger.info(f"  {key}: <tensor shape={value.shape}>")
            else:
                logger.info(f"  {key}: {value}")
    logger.info("=" * 50)

    simulation.attach_model_and_configurations(
        model, initial_data_list, beta=betas
    )

    # Reset memory stats for accurate measurement
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Record timing for throughput measurement
    start_time = time.perf_counter()

    if profile_output:
        # Use profile_start_step/profile_end_step from simulation config
        start_step = simulation.profile_start_step or 0
        end_step = simulation.profile_end_step or simulation.n_timesteps
        n_profile_steps = end_step - start_step + 1

        logger.info(f"Torch profiler enabled, output: {profile_output}")
        logger.info(
            f"  Profiling steps {start_step} to {end_step} "
            f"({n_profile_steps} steps)"
        )

        # Schedule: wait until start_step, then profile continuously
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(
                wait=start_step,
                warmup=min(10, n_profile_steps // 10),
                active=n_profile_steps,
                repeat=1,
            ),
            on_trace_ready=tensorboard_trace_handler(profile_output),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            simulation.simulate(prof=prof)
        # Note: tensorboard_trace_handler already saves Chrome-compatible traces
        # (.pt.trace.json files can be loaded in chrome://tracing)
        logger.info(f"Profiler trace saved to {profile_output}/")
    else:
        simulation.simulate()

    end_time = time.perf_counter()

    # Calculate and log throughput metrics
    elapsed_time = end_time - start_time
    n_timesteps = simulation.n_timesteps
    n_sims = simulation.n_sims
    n_atoms = simulation.n_atoms

    # Get post-warmup throughput metrics (skips first half for accurate measurement)
    throughput_metrics = simulation.get_throughput_metrics()

    # Get peak memory usage (total, for backwards compatibility)
    if torch.cuda.is_available():
        total_max_memory_allocated = torch.cuda.max_memory_allocated() / (1024**3)  # GB
        total_max_memory_reserved = torch.cuda.max_memory_reserved() / (1024**3)  # GB
    else:
        total_max_memory_allocated = 0
        total_max_memory_reserved = 0

    logger.info("=" * 50)
    logger.info("Throughput Metrics (Second Half)")
    logger.info("-" * 50)
    logger.info(f"Total wall time: {elapsed_time:.2f} s")
    logger.info(f"Timesteps: {n_timesteps}, Molecules: {n_sims}, Atoms/mol: {n_atoms}")

    if throughput_metrics:
        first_half = throughput_metrics["first_half_steps"]
        second_half_steps = throughput_metrics["second_half_steps"]
        second_half_time = throughput_metrics["second_half_elapsed_time"]
        throughput = throughput_metrics["throughput"]
        ms_per_timestep = throughput_metrics["ms_per_timestep"]
        peak_mem_alloc = throughput_metrics.get("peak_memory_allocated_gb", 0)
        peak_mem_reserved = throughput_metrics.get("peak_memory_reserved_gb", 0)

        logger.info(f"Measured: second half ({second_half_steps} steps) in {second_half_time:.2f} s")
        logger.info(f"Throughput: {throughput:.2f} (timesteps*molecules)/sec")
        logger.info(f"ms/timestep: {ms_per_timestep:.3f}")
        logger.info(f"Peak memory allocated (2nd half): {peak_mem_alloc:.2f} GB")
        logger.info(f"Peak memory reserved (2nd half): {peak_mem_reserved:.2f} GB")
    else:
        # Fallback to total throughput if metrics unavailable
        throughput = (n_timesteps * n_sims) / elapsed_time
        ms_per_timestep = (elapsed_time / n_timesteps) * 1000
        logger.info(f"Throughput (total): {throughput:.2f} (timesteps*molecules)/sec")
        logger.info(f"ms/timestep (total): {ms_per_timestep:.3f}")
        logger.info(f"Peak memory allocated (total): {total_max_memory_allocated:.2f} GB")
        logger.info(f"Peak memory reserved (total): {total_max_memory_reserved:.2f} GB")

    logger.info("=" * 50)

    logger.info("Simulation completed")


if __name__ == "__main__":
    main()
