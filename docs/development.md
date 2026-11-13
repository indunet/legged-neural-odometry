**English** | [简体中文](zh-CN/development.md)

# Development and reproduction

## Environment

```bash
bash scripts/setup.sh            # Python 3.12, uv, Linux; CPU-only PyTorch by default, no NVIDIA driver needed
sudo apt install libosmesa6      # MuJoCo offscreen rendering without a GPU
.venv/bin/python -m g1_odom fetch --verify
```

With an NVIDIA GPU and a matching driver use `bash scripts/setup.sh --cuda` to install the CUDA build of PyTorch.
Dependencies are declared only in `pyproject.toml`; the script installs this package in editable mode
(`pip install -e ".[dev]"`) and verifies `assets/unitree_g1/`. The project uses a `src/` layout and subprocesses are
started with `python -m g1_odom...`, so the editable install must be done first.
Optional dependency groups in `pyproject.toml`: `sim` (MuJoCo simulation), `train` (PyTorch/ONNX training) and `dev`
(tests and browser smoke test).

The G1 model and walking policy live in `assets/unitree_g1/`; their origin and checksums are recorded in
`configs/assets.lock.json`. `scripts/fetch_assets.py` only verifies; it does not clone external repositories. The
simulation loads these files with plain MuJoCo and does not install mjlab.

Local resource configuration: copy `configs/resources.example.json` to `configs/resources.json` (ignored by Git).
`device` defaults to `cpu` and `render_gl` to `osmesa`. With a GPU you can set `render_gl` to `egl` and set
`render_gpu`. `render_width`, `render_height` and `render_fps` (default 15, wall-clock) control the live web picture;
`render_shadows` (default `false`) turns on shadows and 4x MSAA, which costs about 250 ms per frame under `osmesa` and
should only be enabled with `egl`. An already set `MUJOCO_GL` environment variable takes precedence over the config file.

## Reproducing the full pipeline

```bash
# collect (existing matching trajectories are reused) -> data audit -> label audit -> train -> calibrate -> test;
# every stage can be interrupted and resumed
.venv/bin/python -m g1_odom run --device cpu
# run a single stage
.venv/bin/python -m g1_odom run --stage evaluate --skip-online
```

The stages are `collect`, `audit`, `labels`, `train`, `calibrate` and `evaluate`. Data goes to `data/g1_planar/`,
checkpoints and logs to `runs/`, reports to `reports/`. Training only accepts data under `data/g1_planar/`; checkpoints
record the hashes of labels, trainer and training data and refuse reuse when they differ, so use a fresh run directory.
Sampling uses at most 8 processes by default; lower it with `workers` in `resources.json` or `--workers`.
A full retrain takes about 15 minutes on a GPU; the same pipeline is much slower on CPU, and numbers will not match GPU
training bit for bit.

> The content hashes of `src/g1_odom/labels.py` and `trainer.py` are written into checkpoints and `releases/model.json`.
> Editing these two files, even only formatting, invalidates existing checkpoints.

## Tests

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest
PLAYWRIGHT_BROWSERS_PATH="$PWD/.cache/browsers" .venv/bin/python scripts/web_smoke.py
```

The browser smoke test needs Playwright and its Chromium:
`PLAYWRIGHT_BROWSERS_PATH="$PWD/.cache/browsers" .venv/bin/playwright install chromium`.
Screenshots are written to `docs/images/` and results to `reports/web_smoke.json`.
Some tests need `assets/unitree_g1/` (`fetch --verify`) or `data/g1_planar/`; tests without data are skipped.

## Release

The only GitHub Actions workflow is `.github/workflows/release.yml`. When a GitHub Release (tag `vX.Y.Z`) is published
it runs `scripts/build-deb.sh` on an amd64 and an arm64 runner inside a ROS 2 Humble container and attaches the debs,
their `.sha256` files, `estimator.onnx` and `calibration.json` to that release. The tag must match the versions in
`pyproject.toml`, `Cargo.toml` and `package.xml`. To package locally run `bash scripts/build-deb.sh`; the output goes to
`dist/` (needs ROS 2 Humble, Rust, `fakeroot` and ONNX Runtime 1.31). See [CONTRIBUTING.md](../CONTRIBUTING.md).

## Git

Training data, dependencies, checkpoints and run files are ignored by Git; `assets/unitree_g1/`, released ONNX files,
source, configs and reports can be committed.
