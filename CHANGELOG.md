# Changelog

## 1.0.0

- ROS 2 Humble node `legged_neural_odometry` (Rust, `r2r` + `ort`): subscribes to `/lowstate`, publishes `/g1/odom`
  with pose covariance and a `reset` service. Bundled message definitions live in `ros2/unitree_hg`.
- Debian packaging: `scripts/build-deb.sh` builds `dist/legged-neural-odometry_<version>_<arch>.deb`
  (installs to `/opt/legged-neural-odometry`, enables a systemd service).
- GitHub Actions: the only workflow builds the `amd64` and `arm64` debs when a GitHub Release is published and
  attaches them (plus checksums and the ONNX model) to that release. The pytest CI workflow was removed.
- Environment: dependencies are declared only in `pyproject.toml`; `scripts/setup.sh [--cuda]` replaces
  `bootstrap.sh`, `bootstrap-cpu.sh`, `requirements-cpu.txt` and `requirements-lock.txt`.
- Documentation is bilingual: English by default, Chinese in `README.zh-CN.md` and `docs/zh-CN/`.
- Live web picture no longer stutters under CPU rendering: shadows and MSAA are off by default (`render_shadows`),
  frames are paced by wall clock at up to 15 fps and a frame is only rendered once the simulation has caught up with
  real time, so a slow renderer lowers the frame rate instead of slowing the robot down. Telemetry sends the last 15 s of samples plus a thinned full-run trajectory (`track`) instead of 10000 rows.

## 0.1.0

- Fix the simulation policy adapter: with a zero velocity command (norm < 0.1) the gait phase is zeroed and the robot
  stands still, matching the Unitree deployment code. Previously the robot stepped in place and drifted.
  Data, models, calibration and reports were all regenerated.
- The "Odometry" page now uses an MJPEG video stream background (1280x720, 30 fps) with floating panels and a 1 m
  world grid on the ground so that motion is visible.
- License: Apache License 2.0.
- First cleanup: `src/` layout, `pyproject.toml`, the `g1-odom` command, `docs/`.
- Planar odometry release: 3-seed TCN ensemble with aleatoric / epistemic uncertainty, pose covariance and `quality` alarms.
