#!/usr/bin/env bash
# Create .venv, install the project with all extras and verify the vendored G1 assets.
#
#   bash scripts/setup.sh          # CPU-only PyTorch (default, no NVIDIA / CUDA packages)
#   bash scripts/setup.sh --cuda   # default PyTorch wheel from PyPI (CUDA build)
#
# Requires Linux, uv and Python 3.12. Dependencies are declared in pyproject.toml.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

TORCH_VERSION="2.11.0"
CUDA=0
for arg in "$@"; do
  case "${arg}" in
    --cuda) CUDA=1 ;;
    -h|--help) sed -n '2,8p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "setup: unknown option ${arg}" >&2; exit 2 ;;
  esac
done

if ! command -v uv >/dev/null 2>&1; then
  echo "setup: uv is required (https://docs.astral.sh/uv/)" >&2
  exit 1
fi

if [ ! -x .venv/bin/python ]; then uv venv --python 3.12 .venv; fi
mkdir -p runs outputs data

if [ "${CUDA}" -eq 1 ]; then
  uv pip install --python .venv/bin/python "torch==${TORCH_VERSION}"
else
  uv pip install --python .venv/bin/python "torch==${TORCH_VERSION}" \
    --index-url https://download.pytorch.org/whl/cpu
fi
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python scripts/fetch_assets.py
echo "Done. Activate with: source .venv/bin/activate"
