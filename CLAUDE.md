# CLAUDE.md

## Project Overview

FlashMD is a lightweight Python package for GPU-accelerated coarse-grained molecular dynamics simulation using Triton-optimized SchNet neural network force fields.

## Common Commands

```bash
# Install
pip install -e .

# Test
pytest                    # All tests
pytest --light            # Skip heavy tests
pytest tests/models/      # Specific directory

# Format
black . --line-length 80

# Run simulation (all optimizations ON by default)
flashmd-langevin --config examples/langevin.yaml \
  --model_file model.pt --structure_file configs.pt --betas '[1.67]'

# Run without optimizations (baseline)
flashmd-langevin --disable_optim --config examples/langevin.yaml \
  --model_file model.pt --structure_file configs.pt --betas '[1.67]'
```

## Source Layout (`src/flashmd/`)

- **`models/`** - SchNet architecture (`schnet.py`), MLP, cutoff, gradients (GradientsOut/SumOut), GPTQ quantization, radial basis
- **`prior/`** - Physical prior terms (harmonic bonds/angles, repulsion, dihedrals, polynomials)
- **`kernels/`** - Triton GPU kernels: `cfconv_kernels.py` (fused CFConv), `csr_kernels.py` (CSR segment reduce)
- **`simulation/`** - MD integrators: Langevin (BAOAB), parallel tempering, velocity Verlet
- **`data/`** - `AtomicData` core data structure
- **`geometry/`** - Internal coordinate computations (distances, angles, torsions)
- **`neighbor_list/`** - Neighbor list building
- **`hub.py`** - HuggingFace model loading
- **`scripts/`** - CLI entry points (`flashmd-langevin`, `flashmd-pt-langevin`, `flashmd-nve-verlet`)

## Environment Variables

All Triton optimizations are **ON by default**. Use `--disable_optim` CLI flag to turn them all off, or set individual variables to `"0"`.

| Variable | Default | Effect |
|----------|---------|--------|
| `MLCG_USE_TRITON_MESSAGE_PASSING` | `1` | Fused cutoff+gather+multiply+scatter |
| `MLCG_USE_FUSED_RBF` | `1` | Fused distance+RBF+cutoff |
| `MLCG_USE_CSR` | `1` | CSR segment reduce for scatter |
| `MLCG_USE_SRC_CSR_GRAD_X` | `1` | Source-CSR for backward grad computation |
| `MLCG_USE_FUSED_TANH_LINEAR` | `1` | Fused tanh+linear in interaction blocks |

Simulation defaults: `compile_model=True`, `gptq="w16a16"`. Both disabled by `--disable_optim`.

## Rules

- Use Linus' coding style. Always use English in code files.
- Do not modify number of timesteps when running benchmarks unless explicitly asked.
- When benchmarking, only use GPUs with completely empty memory/utilization.
- Simulation dynamics may only stabilize after 90k+ steps.
- Use color palette `#D87756`, `#689BCC`, `#C46686` for plots, Nature camera-ready standard. Save PDF only.
