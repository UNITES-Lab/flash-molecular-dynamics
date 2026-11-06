# FlashMD unification

Status: in progress on branch `unify`. Started 2026-11-05.

FlashMD runs published learned-simulation checkpoints at ensemble scale on one
GPU. This document tracks the merge of FlashMACE and FlashBioEmu into the
FlashSchNet package.

## Goal

One package with a vLLM-style runtime and Hugging Face-style model loading:

- `flashmd.load(name)` returns a model built from published weights, unchanged.
- `Simulator` and `Sampler` batch replicas or samples, size batches to GPU
  memory, replay CUDA graphs, and write resumable outputs.
- Each architecture is a plugin with a reference path (the unmodified upstream
  module) and a fast path. `flashmd.verify` compares them on the user's GPU.

## Principles

1. **Wrap, never fork.** MACE and BioEmu come from pinned upstream packages
   (`mace-torch>=0.3.16,<1.0`, `bioemu==1.4.1`). FlashMD swaps execution from
   outside through module swaps and wrappers over unchanged parameters. The
   audit found no change that requires editing upstream.
2. **Every fast path has a reference twin** and a parity test.
3. **One configuration object.** `ExecutionConfig` replaces the import-time
   `MLCG_USE_*` variables, `MACE_DISABLE_TRITON_GEOMETRY`, and scattered flags.
   Strict precision is the default.
4. **The engine owns the loop.** Plugins map positions to energies and forces,
   or a sequence to samples. Batching, neighbor lists, integrators, graphs,
   memory and output are shared.
5. **Google Python style, enforced by tools** (see `CONTRIBUTING.md`).

## Architecture

```
interface   flashmd.load · Simulator · Sampler · verify · flashmd CLI
engine      replica batching, memory planner, CUDA-graph runner, precision,
            integrators, replica exchange, neighbor lists, DPM loop, writers
plugins     CGSchNet (vendored mlcg lineage) · MACE (adapter over mace-torch)
            · BioEmu (adapter over bioemu)
kernels     shared graph kernels (CSR, segment reduce, edge geometry, fp16 GEMM)
            and per-model kernels
upstream    mace-torch, bioemu (pinned, unmodified); weights from the HF Hub
```

## Plugin contract

```python
class Potential(abc.ABC):
    def load(cls, source, *, device, dtype) -> Potential: ...
    def optimize(self, config: ExecutionConfig) -> None: ...
    def prepare(self, systems) -> PreparedBatch: ...
    def compute(self, batch, positions) -> PotentialOutput: ...

class Generator(abc.ABC):
    def load(cls, source, *, device, dtype) -> Generator: ...
    def optimize(self, config: ExecutionConfig) -> None: ...
    def sample(self, request, *, generator) -> Iterator[SampleBatch]: ...
```

The concrete definitions live in `src/flashmd/models/base.py`.

## Correctness gates

Found by reading the code and by CPU emulation. Each gets a GPU test in phase
P1 that confirms and fixes it, or refutes it.

| Model | Finding | Status |
|---|---|---|
| BioEmu | Triton IPA kernels are always on and read `R` as dense; on the plain GPU path `R` is a transposed view, so samples are wrong. Protein-size baseline and production runs did not use that path. | open |
| BioEmu | Upstream never calls `.eval()` and released configs set `dropout: 0.1`; FlashBioEmu runs in eval mode. | open |
| BioEmu | Static context cache skips the per-layer pair update of v1.2 (`extra_residue_embeds`). | open |
| BioEmu | Accelerated path ignores the configured denoiser (steering, Heun). | open |
| MACE | The fp64 to fp32 cast is 6.5x of the gain over the cuEq float64 baseline; a precision-matched arm is needed. | open |
| MACE | Benchmarks used MACE-MP-0, vacuum, and a 5-bead CG structure in the all-atom model; the benchmark loop is a damped relaxation. | open |
| MACE | Global Triton geometry hook drops the `shifts` gradient and is not double-differentiable. | open |
| MACE | `FixedOneHotFCTP` reuses primed element groups without checking composition. | open |
| CGSchNet | Fused CFConv autograd returns `None` for `edge_weight`; the cutoff-envelope derivative may be missing from forces. | open |
| CGSchNet | `--disable_optim` does not disable `torch.compile`: the scripts set `simulation.compile_model`, but `_attach_model` reads `_compile_model_flag`. The Langevin and Verlet scripts also enable TF32 matmuls unconditionally, so the CLI reference arm was not strict float32. | fixed on `unify` (property plus TF32 only with optimizations); check whether any published baseline used these scripts |
| CGSchNet | `fused_src_csr_grad_x` is wrong for `num_filters > 128`. | open |
| CGSchNet | The fused RBF backward uses `math.pi` without importing `math`, so any `GaussianBasis` model with `MLCG_USE_FUSED_RBF=1` would raise `NameError` when forces are computed. The public checkpoint pickles `ExpNormalBasis`, so the fused RBF path may never run for it; reconcile with the 1.7x Flash RBF step in the FlashSchNet ablation. | open |
| CGSchNet | The public checkpoint pickles `mlcg.nn.*` classes; loading needs mlcg or a class remap. | fixed in P0: `models/legacy_pickle.py`; CPU parity with upstream mlcg in `benchmarks/parity/cgschnet_cpu` |
| CGSchNet | Kernels that run for the public checkpoint (checked on the loaded model): the Triton CFConv is on in all 4 blocks; the fused tanh-linear kernel is off in all 4, because the pickle lacks `use_fused_tanh_linear` and both FlashMD and the fork patch it to False; the fused RBF kernel never runs, because the model uses `ExpNormalBasis`; 128 filters, so the source-CSR backward bug is not reached. Reconcile with the FlashSchNet ablation steps. | open |
| CGSchNet | `gptq.py` is a plain fp16 cast; `MLCG_USE_FUSED_RBF` values other than 0 or 1 raise `NameError`; resume picks the init checkpoint. | open |

## Target layout

```
src/flashmd/
  config.py  registry.py  hub.py
  core/       data, neighbors, precision, graphs, memory, io
  engine/     simulator, integrators, replica_exchange, sampler
  kernels/    shared Triton kernels
  models/     base.py, schnet/, mace/, bioemu/
  cli/        flashmd simulate | sample | bench | verify | convert
  _legacy/    old import paths and command aliases
benchmarks/   protein-size scaling and per-model harnesses
```

## Phases

### P0 Groundwork

- [x] `unify` branch, Google-style tooling (ruff, mypy, pre-commit), CPU CI
- [x] `THIRD_PARTY_NOTICES.md` with the full license texts
- [x] Benchmark harness moved from the server into `benchmarks/legacy`
- [ ] Locked environment with all three upstreams on the cluster
- [ ] Golden outputs from the current code on fixed inputs
- [x] Public package loads the published CGSchNet checkpoint without mlcg
  (`ExpNormalBasis` port, class remap, CPU parity with upstream mlcg)

### P1 Correctness gates

- [ ] One GPU test per gate; fix what is confirmed
- [ ] Rerun affected benchmarks; before and after table
- [ ] BioEmu train-mode versus eval-mode pocket probability on one system

### P2 Core and the CGSchNet plugin

- [ ] `core/` and `engine/` extracted; `ExecutionConfig` replaces env vars
- [ ] CGSchNet plugin; legacy import paths and commands still work
- [ ] CUDA-graph replay for the SchNet step
- Exit: force parity at fp32 tolerance; 1ENH at 64 replicas within 3% of
  golden throughput

### P3 MACE plugin

- [ ] Adapter over stock `mace-torch` 0.3.17
- [ ] All-atom inputs; MACE-OFF and MACE-MP-0 in the registry
- [ ] Precision-matched baseline arms
- Exit: parity against unmodified upstream at fp32

### P4 BioEmu plugin

- [ ] `FlashDiG` wrapper, context cache, deterministic DPM loop, graph runner
- [ ] Fixed IPA kernels; explicit eval policy; denoiser guard; v1.2 refused
- Exit: per-step and final parity against upstream `dpm_solver` under
  identical noise

### P5 Unified API, documentation, release

- [ ] `load`, `Simulator`, `Sampler`, `verify`; `flashmd` CLI; memory planner
- [ ] Documentation site, examples, reviewer package
- [ ] PyPI as `flash-molecular-dynamics`; Zenodo DOI
- Exit: one command reproduces the protein-size figure on a clean machine

## Decisions

| Decision | Choice |
|---|---|
| Where development happens | Public repository, `unify` branch, never `main` until release |
| BioEmu reference | Eval mode, documented; P1 quantifies the difference |
| Import name | Keep `flashmd`; ship as `flash-molecular-dynamics` (PyPI `flashmd` belongs to lab-cosmo's FlashMD) |
| Scope | CGSchNet, MACE, BioEmu; FlashSchake and a CUDA kernel backend later |
