#!/usr/bin/env bash
# Regenerates requirements/ci-cpu.lock.txt, the pinned CPU environment CI
# installs. Run after changing dependencies in pyproject.toml.
set -euo pipefail
cd "$(dirname "$0")/../.."
TORCH=$(sed -n 's/^torch==//p' requirements/ci-cpu.constraints.txt)
uv pip compile pyproject.toml --extra dev \
    --constraint requirements/ci-cpu.constraints.txt \
    --python-version 3.12 --python-platform x86_64-unknown-linux-gnu \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    --find-links "https://data.pyg.org/whl/torch-${TORCH}+cpu.html" \
    --index-strategy unsafe-best-match \
    --emit-index-url --emit-find-links --no-header \
    --output-file requirements/ci-cpu.lock.txt
