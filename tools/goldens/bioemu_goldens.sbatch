#!/bin/bash
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH -x unites3.ib,unites4.ib,unites5.ib
#SBATCH --time=00:40:00
#SBATCH -J fmd-golden-bioemu
#SBATCH -o /playpen-shared/pingzhi/efficient-schnet/unify-work/goldens/bioemu/logs/slurm-%j.out
#
# Golden BioEmu outputs of the official tree and the FlashBioEmu fork for
# ubiquitin, bioemu-v1.1, batch 32, 9 DPM steps (see bioemu_goldens.py).
# Writes only under $OUT: inputs are copied there, and the Triton, CUDA and
# bytecode caches are redirected or disabled.
set -euo pipefail

ROOT=/playpen-shared/pingzhi/efficient-schnet
OUT="$ROOT/unify-work/goldens/bioemu"
PY=/playpen-shared/pingzhi/anaconda3/envs/bioemu/bin/python
OFFICIAL="$ROOT/bioemu-official-benchmark"
FLASH="$ROOT/flash-bioemu"
SNAPSHOT=/playpen-shared/pingzhi/hf-cache/hub/models--microsoft--bioemu/snapshots/1fec9d4473a9da04f5e60026a70d7a93fb893e9e
CKPT_DIR="$SNAPSHOT/checkpoints/bioemu-v1.1"
SCALING="$ROOT/submission/data/protein_size_scaling"
FASTA="$SCALING/inputs/bioemu/ubiquitin_76aa.fasta"
UBQ_SHA=233b4b0b8c4616095bc3249f9375fc345d32af7deaef07117d0383c51d6f19aa

export JAX_PLATFORM_NAME=cpu
export PYTHONDONTWRITEBYTECODE=1
export TRITON_CACHE_DIR="$OUT/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$OUT/cache/inductor"
export CUDA_CACHE_PATH="$OUT/cache/nv"
export MPLCONFIGDIR="$OUT/cache/matplotlib"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
mkdir -p "$OUT/logs" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
  "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" "$OUT/inputs/embed_cache" \
  "$OUT/inputs/so3_cache"
cd "$OUT"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none}" \
  "cuda_visible=${CUDA_VISIBLE_DEVICES:-unset} start=$(date -Is)"
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,uuid --format=csv \
  | tee "$OUT/logs/nvidia-smi-${SLURM_JOB_ID:-local}.csv"

# Private copies of the paper's inputs, so no run writes next to them.
cp -p "$SCALING/bioemu_embed_cache/$UBQ_SHA"* "$OUT/inputs/embed_cache/"
cp -p "$FLASH/profiles/so3_cache/"*.npz "$OUT/inputs/so3_cache/"
sha256sum "$OUT/inputs/embed_cache/"* "$OUT/inputs/so3_cache/"*

common=(
  --out-dir "$OUT"
  --checkpoint "$CKPT_DIR/checkpoint.ckpt"
  --model-config "$CKPT_DIR/config.yaml"
  --fasta "$FASTA"
  --embeds-dir "$OUT/inputs/embed_cache"
  --so3-cache-dir "$OUT/inputs/so3_cache"
  --batch-size 32
  --steps 9
  --seed 20260616
)

PYTHONPATH="$OFFICIAL/src" "$PY" -B "$OUT/bioemu_goldens.py" run \
  --tree official --tree-root "$OFFICIAL" "${common[@]}"
PYTHONPATH="$FLASH/src" "$PY" -B "$OUT/bioemu_goldens.py" run \
  --tree flash --tree-root "$FLASH" "${common[@]}"
env -u PYTHONPATH "$PY" -B "$OUT/bioemu_goldens.py" compare --out-dir "$OUT"
echo "done=$(date -Is)"
