# Contributing

Thank you for your interest! This file is in English; the project README is also available in
[Chinese](README.zh-CN.md).

## Development environment

```bash
bash scripts/setup.sh              # CPU-only PyTorch; use --cuda for the CUDA build
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest
```

Dependencies are declared only in `pyproject.toml` (extras `sim`, `train`, `dev`). See
[docs/development.md](docs/development.md) for the full workflow.

## Making changes

- Run `pytest` before submitting. When you change the web page, also run `scripts/web_smoke.py`.
- The inference path (`runtime.py`, `sensors.py`, `uncertainty.py`, `standstill.py`) must not depend on `torch` or
  `mujoco`; the tests check this.
- Do not modify `labels.py` or `trainer.py` unless you intend to retrain and re-release the model: their hashes are
  stored in checkpoints and in the release metadata.
- Do not commit data, checkpoints (`*.pt`), logs or build output. `assets/unitree_g1/` ships with the repository;
  the released odometry ONNX files live in `releases/`.
- After changing the model, labels or evaluation, rerun the full pipeline and update `releases/` and `reports/`.
- By contributing you agree to license your contribution under the [Apache License 2.0](LICENSE).
- When reporting a problem, include the Python, PyTorch and MuJoCo versions and the command that reproduces it.

## Cutting a release

Releases are built by GitHub Actions; there is no other CI workflow.

1. Update the version everywhere it appears: `pyproject.toml`, `ros2/legged_neural_odometry/Cargo.toml`
   (the build uses `--locked`, so run `cargo build` once in that directory to refresh `Cargo.lock`),
   `ros2/legged_neural_odometry/package.xml`, `ros2/unitree_hg/package.xml`. Add an entry to `CHANGELOG.md`.
2. Merge to the default branch, then publish a GitHub Release with the tag `vX.Y.Z`.
3. `.github/workflows/release.yml` builds `legged-neural-odometry_X.Y.Z_amd64.deb` and `..._arm64.deb` in a
   ROS 2 Humble container, and attaches them, their `.sha256` files and `estimator.onnx` / `calibration.json`
   to the release. The build fails if the tag does not match the versions above.
4. To test packaging without publishing, run the workflow manually (Actions > release > Run workflow) or run
   `bash scripts/build-deb.sh` locally (needs ROS 2 Humble, Rust, `fakeroot`, and ONNX Runtime 1.31 via
   `ORT_DYLIB_PATH`); the package is written to `dist/`.
