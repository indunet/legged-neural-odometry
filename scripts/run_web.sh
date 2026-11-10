#!/usr/bin/env bash
# Start the local Web console (default http://127.0.0.1:18078).
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
exec .venv/bin/python -m g1_odom serve "$@"
