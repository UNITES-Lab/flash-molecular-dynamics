#!/usr/bin/env bash
#SBATCH -J fmd-gates-cgschnet
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH -x unites3.ib,unites4.ib,unites5.ib
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -o %x-%j.out
# Runs every CGSchNet gate arm on one GPU, then the comparison.
# Required environment: PYTHON, REPO, DATA (checkpoint and 1ENH configs), OUT.
set -euo pipefail
: "${PYTHON:?}" "${REPO:?}" "${DATA:?}" "${OUT:?}"

mkdir -p "$OUT"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
git -C "$REPO" rev-parse HEAD
export PYTHONPATH="$REPO/src"
export TRITON_CACHE_DIR="$OUT/.triton"
GATES="$REPO/tools/gates/cgschnet_gpu_gates.py"

for model in public gauss128 exp256; do
    for arm in reference_fp64 reference reference_detached_envelope \
               flash flash_w16a16; do
        "$PYTHON" -W ignore "$GATES" run --arm "$arm" --model "$model" \
            --checkpoint "$DATA/model_and_prior.pt" \
            --configurations "$DATA/1enh_configurations.pt" \
            --out "$OUT" || echo "process for $model/$arm exited with $?"
    done
done
for model in public gauss128 exp256; do
    for switch in MLCG_USE_TRITON_MESSAGE_PASSING MLCG_USE_FUSED_RBF \
                  MLCG_USE_FUSED_TANH_LINEAR MLCG_USE_CSR \
                  MLCG_USE_SRC_CSR_GRAD_X; do
        for mode in only without; do
            "$PYTHON" -W ignore "$GATES" run --arm "$mode:$switch" \
                --model "$model" \
                --checkpoint "$DATA/model_and_prior.pt" \
                --configurations "$DATA/1enh_configurations.pt" \
                --out "$OUT" || echo "process for $model/$mode:$switch exited"
        done
    done
done
"$PYTHON" "$GATES" compare --out "$OUT"
