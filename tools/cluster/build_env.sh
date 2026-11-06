#!/usr/bin/env bash
# Builds the unified FlashMD environment for goldens and GPU tests.
#
# Usage: tools/cluster/build_env.sh PREFIX [LOCK_OUT]
#
# Creates a conda prefix with Python 3.12, installs torch from the CUDA 12.8
# index, torch_cluster from the PyG wheel index, then this repository with
# the MACE and BioEmu extras. Writes `pip freeze` to LOCK_OUT when given.
set -euo pipefail

PREFIX=${1:?usage: build_env.sh PREFIX [LOCK_OUT]}
LOCK_OUT=${2:-}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
CONDA=${CONDA:-conda}
PYTHON_VERSION=3.12
TORCH_VERSION=2.10.0
CUDA_TAG=cu128

"$CONDA" create -y -q -p "$PREFIX" "python=$PYTHON_VERSION" pip
PY="$PREFIX/bin/python"
PIP=("$PY" -m pip install -q --root-user-action=ignore)

CONSTRAINTS="$PREFIX/constraints.txt"
echo "torch==$TORCH_VERSION" > "$CONSTRAINTS"

"${PIP[@]}" --upgrade pip
"${PIP[@]}" "torch==$TORCH_VERSION" \
    --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
"${PIP[@]}" torch_cluster -c "$CONSTRAINTS" \
    -f "https://data.pyg.org/whl/torch-$TORCH_VERSION+$CUDA_TAG.html"
"${PIP[@]}" -c "$CONSTRAINTS" -e "$REPO[all,mace-cuda12,dev]"

"$PY" - <<'PY'
import importlib.metadata as md
for name in ("torch", "triton", "torch_cluster", "torch_geometric",
             "mace-torch", "e3nn", "cuequivariance-torch", "bioemu", "numpy"):
    try:
        print(f"{name}=={md.version(name)}")
    except md.PackageNotFoundError:
        print(f"{name} MISSING")
PY

if [[ -n "$LOCK_OUT" ]]; then
    "$PY" -m pip freeze --exclude-editable > "$LOCK_OUT"
fi
