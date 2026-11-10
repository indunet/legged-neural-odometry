#!/usr/bin/env bash
# Create .venv with the locked dependencies, install this package in editable mode and fetch pinned assets.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [ ! -x .venv/bin/python ]; then uv venv --python 3.12 .venv; fi
mkdir -p runs outputs data releases
uv pip install --python .venv/bin/python --link-mode copy -r requirements-lock.txt
uv pip install --python .venv/bin/python --no-deps -e .
.venv/bin/python scripts/fetch_assets.py
