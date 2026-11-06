# Protein-size scaling benchmarks

These data support joint protein-size and batch-size scaling at batch sizes
1, 2, 4, 8, 16, 32, and 64, as well as the standalone fixed-batch figures at
1 and 64. All measurements were collected on one NVIDIA RTX PRO 6000 Blackwell
Server Edition GPU (97,887 MiB, driver 595.71.05).

## Measurements

- **CGSchNet:** 1,000 fixed-conformation force steps (`dt: 0`), with the second
  500 steps timed and trajectory/logging I/O disabled. Throughput is replica
  steps per second. The baseline disables the fused paths; FlashMD enables the
  Triton message-passing, fused-RBF, CSR, source-gradient, and W16A16 paths.
- **MACE:** 12 fixed-geometry energy/force evaluations, with the last 8 timed.
  Throughput is replica force evaluations per second. The baseline is CuEq
  float64; FlashMD uses the Triton float32 backend, TF32, compiled core modules,
  fixed one-hot tensor products, and the energy/force-only path.
- **BioEmu:** cached nine-step DPM sampling, with 2 baseline or 5 FlashMD warmup
  runs followed by 5 timed runs. Throughput is completed nine-step samples per
  second. The baseline uses the official implementation; FlashMD adds static
  context caching and CUDA-graph replay.

The protein-size axis is a collection of runnable protein inputs, not a
synthetic resizing of one structure. Exact duplicate sizes are aggregated by
the median in the plotting scripts. The 3D figure uses log2 coordinates for
protein and batch size and log10 coordinates for throughput, while displaying
the original values on every tick. Every surface vertex is measured; no fitted
or interpolated values replace a benchmark. A series-colored `x` at the floor
means that configuration did not complete, and the CSV distinguishes `oom`
from other failures.

## Inputs and provenance

`protein_manifest.json` is the source of truth for every plotted input.
CGSchNet includes 15 native systems plus three MACE structures converted with
the MLCG coarse-graining tools. MACE includes its three native structures plus
nine RCSB heavy-atom chains selected by sequence match to CGSchNet inputs.
BioEmu includes 18 sequence inputs spanning 17 lengths shared from the other
families.

Native BioEmu inputs use their real cached AlphaFold embeddings. Cross-model
sequences use deterministic T4-tiled tensors with the exact target shapes.
Those shape-equivalent inputs measure execution scaling only and must not be
used for biological sample analysis.

At batch size 64, the largest successful baseline/FlashMD sizes are 508/927
beads for CGSchNet and 816/1,166 atoms for MACE; both BioEmu series complete
through 377 residues. The 1,851-bead CGSchNet FlashMD point reproducibly fails
in the fused CSR Triton kernel with an illegal-memory-access error. It is stored
as `failed`, not `oom`, and no throughput is assigned to it.

## Artifacts

- `protein_size_scaling_bs{1,2,4,8,16,32,64}.csv`: seven fixed-batch tables,
  each with 96 rows.
- `protein_batch_scaling.csv`: the combined 672-row long-form table. It contains
  657 successful measurements, 14 OOM records, and one non-OOM failure.
- `results/` and `results_bs{2,4,8,16,32,64}/`: raw logs, metrics, and failure
  records.
- `../../figures/protein_size_scaling_bs1.{png,pdf}` and
  `../../figures/protein_size_scaling_bs64.{png,pdf}`: standalone figures.
- `../../figures/protein_batch_scaling_3d.{png,pdf}`: the final three-panel 3D
  scaling figure. The default camera uses 20-degree elevation and -62-degree
  azimuth to emphasize throughput growth along the batch axis.
- `../../figures/protein_batch_scaling_interactive.html`: a self-contained
  interactive version with synchronized cameras, animated batch slices,
  searchable benchmark rows, and all 672 records embedded in the file.

Across systems that complete at both endpoints, the median FlashMD throughput
gain from batch 1 to 64 is 12.9x for CGSchNet, 13.1x for MACE, and 8.5x for
BioEmu.

Regenerate the tables and figures from the repository root with:

```bash
for batch_size in 1 2 4 8 16 32 64; do
  python submission/scripts/run_protein_size_scaling.py collect --batch-size "$batch_size"
done
python submission/scripts/build_protein_size_scaling_figure.py --batch-size 1
python submission/scripts/build_protein_size_scaling_figure.py --batch-size 64
python submission/scripts/build_protein_batch_scaling_3d_figure.py
python submission/scripts/build_protein_batch_scaling_interactive.py
```
