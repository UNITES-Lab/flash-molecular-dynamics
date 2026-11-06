#!/bin/bash
# Captures FlashMACE golden energies and forces with mace_goldens.py.
#
# Copy mace_goldens.py and this file into GOLDEN_DIR, then submit:
#   sbatch "$GOLDEN_DIR/mace_goldens.sbatch"
# Writes only under GOLDEN_DIR: golden files, summary.json, the Slurm log
# and compiler caches.
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH -x unites3.ib,unites4.ib,unites5.ib
#SBATCH --time=00:40:00
#SBATCH -J fmd-golden-mace
#SBATCH -o /playpen-shared/pingzhi/efficient-schnet/unify-work/goldens/mace/slurm-%j.out

set -euo pipefail

FLASH_MACE=/playpen-shared/pingzhi/efficient-schnet/flash-mace
GOLDEN_DIR=/playpen-shared/pingzhi/efficient-schnet/unify-work/goldens/mace
PYTHON=/playpen-shared/pingzhi/anaconda3/envs/flash-mace/bin/python

# Keep every cache inside GOLDEN_DIR and never write .pyc files into the
# FlashMACE checkout, which must stay untouched.
CACHE="$GOLDEN_DIR/.cache"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor"
export TRITON_CACHE_DIR="$CACHE/triton"
export CUEQ_TRITON_CACHE_DIR="$CACHE/cuequivariance-triton"
export TORCH_EXTENSIONS_DIR="$CACHE/torch_extensions"
export CUDA_CACHE_PATH="$CACHE/nv"
export XDG_CACHE_HOME="$CACHE/xdg"
export PYTHONDONTWRITEBYTECODE=1
# Same import path and torch.load default as the paper's benchmark runs.
export PYTHONPATH="$FLASH_MACE"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
# The arms set this themselves; an inherited value would corrupt arm c.
unset MACE_DISABLE_TRITON_GEOMETRY
# libcue_ops.so (cuequivariance-ops-torch-cu12) needs libcublas.so.12, which
# torch cu130 does not load. If the node's linker cache lacks it, use the
# copy the env installed with cuequivariance (nvidia-cublas-cu12).
SITE=/playpen-shared/pingzhi/anaconda3/envs/flash-mace/lib/python3.12/site-packages
if ! /sbin/ldconfig -p 2>/dev/null | grep "libcublas.so.12 " >/dev/null; then
    export LD_LIBRARY_PATH="$SITE/nvidia/cublas/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "libcublas.so.12 not in linker cache; using $SITE/nvidia/cublas/lib"
fi
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"

cd "$GOLDEN_DIR"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} date=$(date -u +%FT%TZ)"
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo "flash-mace HEAD=$(git --no-optional-locks -C "$FLASH_MACE" rev-parse HEAD)"
"$PYTHON" -c "import sys; print(sys.version)"

"$PYTHON" "$GOLDEN_DIR/mace_goldens.py" \
    --flash-mace "$FLASH_MACE" \
    --out-dir "$GOLDEN_DIR"
