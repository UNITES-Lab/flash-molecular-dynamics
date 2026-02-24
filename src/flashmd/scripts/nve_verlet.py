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

from flashmd.simulation import parse_simulation_config, NVESimulation
from flashmd.simulation.logging import logger, setup_simulation_logging


def main():
    torch.jit.set_fusion_strategy([("DYNAMIC", 3)])
    torch.set_float32_matmul_precision("high")

    disable_overrides = {}
    if _DISABLE_OPTIM:
        disable_overrides = {
            "compile_model": False,
            "gptq": None,
        }

    model, initial_data_list, betas, simulation, profile_output = parse_simulation_config(NVESimulation)

    if disable_overrides:
        for k, v in disable_overrides.items():
            setattr(simulation, k, v)
        logger.info("--disable_optim: compile_model=False, gptq=None")

    setup_simulation_logging(filename=simulation.filename, output_dir=".", add_timestamp=False)

    logger.info(f"Starting NVE (Velocity Verlet) simulation")
    logger.info(f"Simulation filename: {simulation.filename}")

    simulation.attach_model_and_configurations(model, initial_data_list, beta=betas)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_time = time.perf_counter()
    simulation.simulate()
    elapsed = time.perf_counter() - start_time

    throughput = (simulation.n_timesteps * simulation.n_sims) / elapsed
    logger.info(f"Wall time: {elapsed:.2f}s")
    logger.info(f"Throughput: {throughput:.2f} (timesteps*molecules)/sec")

    if torch.cuda.is_available():
        logger.info(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
