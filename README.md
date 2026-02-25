<img src="static/flashmd-logo.png" alt="FlashMD" width="600">

## Flash Molecular Dynamics: High-performance GPU kernels for neural network molecular dynamics simulation.

Preprint: "FlashSchNet: Fast and Accurate Coarse-Grained Neural Network Molecular Dynamics" 

- Author: [Pingzhi Li](https://pingzhili.github.io/), [Hongxuan Li](https://scholar.google.com/citations?user=Cg8IbW8AAAAJ&hl=en), [Zirui Liu](https://zirui-ray-liu.github.io/), [Xingcheng Lin](https://lingroup.wordpress.ncsu.edu/), [Tianlong Chen](https://tianlong-chen.github.io/)
- Paper: [[arXiv]](https://arxiv.org/abs/2602.13140)

FlashMD provides Triton-optimized kernels that accelerate neural network force field evaluation during MD simulation. Currently supports **CGSchNet**; we are actively incorporating more architectures.

<img src="static/flashmd-demo.gif" alt="CGSchNet vs FlashMD simulation comparison" width="600">

## Install

```bash
pip install flashmd
```

Or from source:

```bash
git clone https://github.com/unites-lab/flash-molecular-dynamics.git
cd flash-molecular-dynamics
pip install -e .
```

## Quick start

### Python API

```python
from flashmd.hub import from_pretrained, download_file

# Load pre-trained CGSchNet model from HuggingFace
model = from_pretrained("pingzhili/cg-schnet")

# Download initial configurations
configs_path = download_file("pingzhili/cg-schnet", "1enh_configurations.pt")
```

### Command-line simulation

```bash
# Langevin dynamics
flashmd-langevin --config examples/langevin.yaml \
  --model_file model_and_prior.pt \
  --structure_file 1enh_configurations.pt \
  --betas '[1.67]'

# Parallel tempering
flashmd-pt-langevin --config examples/parallel_tempering.yaml \
  --model_file model_and_prior.pt \
  --structure_file 1enh_configurations.pt \
  --betas '[1.67, 1.42, 1.16]'
```

### Optimizations

All Triton kernel optimizations, `torch.compile`, and W16A16 quantization are **enabled by default**. No environment variables or extra flags needed:

```bash
# All optimizations ON by default
flashmd-langevin --config config.yaml \
  --model_file model.pt --structure_file configs.pt --betas '[1.67]'

# Disable all optimizations (for baseline comparison)
flashmd-langevin --disable_optim --config config.yaml \
  --model_file model.pt --structure_file configs.pt --betas '[1.67]'
```

You can still fine-tune individual kernels via environment variables (`MLCG_USE_TRITON_MESSAGE_PASSING`, `MLCG_USE_FUSED_RBF`, `MLCG_USE_CSR`, `MLCG_USE_SRC_CSR_GRAD_X`, `MLCG_USE_FUSED_TANH_LINEAR`), setting any to `0` to disable.

## Performance

Benchmarked on 1ENH protein (batch size 128, RTX PRO 6000 96fGB GPU):

| Configuration | Throughput (timestep·mol/s) | GPU Memory (GB) | Speedup |
|---|---|---|---|
| CGSchNet | 438 | 94 | 1.0x |
| **FlashMD** | **2861** | **17** | **6.5x** |

Key optimizations under the hood:
- **Fused CFConv kernel**: Merges cutoff, gather, multiply, scatter into one Triton kernel
- **CSR segment reduce**: Replaces atomic scatter with deterministic CSR-based reduction
- **Fused distance+RBF+cutoff**: Single kernel for distance computation, Gaussian RBF expansion, and cosine cutoff
- **Fused tanh+linear**: Fuses activation and linear layer in interaction blocks
- **W16A16**: FP16 quantized filter/output networks using Tensor Cores
- **torch.compile**: Graph-level optimization of the simulation loop

## Pre-trained models

### CGSchNet

Coarse-grained SchNet force field from [Charron et al., *Nat. Chem.* 2025](https://doi.org/10.1038/s41557-025-01874-0).

| Model | HuggingFace | Files |
|---|---|---|
| CGSchNet (1ENH) | [`pingzhili/cg-schnet`](https://huggingface.co/pingzhili/cg-schnet) | `model_and_prior.pt`, `1enh_configurations.pt` |

*More models coming soon.*

## Citation

If you use FlashMD, please cite:

```bibtex
@misc{li2026flashschnet,
  title={FlashSchNet: Fast and Accurate Coarse-Grained Neural Network Molecular Dynamics},
  author={Li, Pingzhi and Li, Hongxuan and Liu, Zirui and Lin, Xingcheng and Chen, Tianlong},
  year={2026},
  eprint={2602.13140},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2602.13140},
}
```

If you use the CGSchNet model, please also cite:

```bibtex
@article{charron2025cgschnet,
  title={Navigating protein landscapes with a machine-learned transferable coarse-grained model},
  author={Charron, Nicholas E. and Bonneau, Klara and Pasos-Trejo, Aldo S. and others},
  journal={Nature Chemistry},
  volume={17},
  pages={1284--1292},
  year={2025},
  doi={10.1038/s41557-025-01874-0}
}
```

## License

MIT
