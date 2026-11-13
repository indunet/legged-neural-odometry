#!/usr/bin/env bash
# CPU-only environment: no NVIDIA / CUDA packages. Needs Python 3.12 and uv.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [ ! -x .venv/bin/python ]; then uv venv --python 3.12 .venv; fi
mkdir -p runs outputs data releases
uv pip install --python .venv/bin/python torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python --link-mode copy -r requirements-cpu.txt
uv pip install --python .venv/bin/python --no-deps -e .
.venv/bin/python scripts/fetch_assets.py
