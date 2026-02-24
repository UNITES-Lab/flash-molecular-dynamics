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

from time import ctime
import os.path as osp
import torch
from typing import Any, Dict


from flashmd.simulation import (
    parse_simulation_config,
    PTSimulation,
)


def main():
    print(f"Starting simulation at {ctime()} with {PTSimulation}")

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
        profile,
    ) = parse_simulation_config(PTSimulation)

    if disable_overrides:
        for k, v in disable_overrides.items():
            setattr(simulation, k, v)
        print("--disable_optim: compile_model=False, gptq=None")

    simulation.attach_model_and_configurations(
        model, initial_data_list, betas=betas
    )
    simulation.simulate()
    print(f"Ending simulation at {ctime()}")


if __name__ == "__main__":
    main()
