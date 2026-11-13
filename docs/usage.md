**English** | [简体中文](zh-CN/usage.md)

# Usage

## Inference API

```python
from g1_odom.runtime import Odometry
from g1_odom.sensors import SensorSample

odom = Odometry()            # default releases/estimator.onnx; or pass a list of members + calibration='releases/calibration.json'
# yaw is the hardware IMU yaw already transformed to the pelvis frame, in rad, e.g. low_state.imu_state.rpy[2]
# (check the mounting frame).
result = odom.update(SensorSample(t, gyro, accel, q, dq, imu_yaw=yaw, imu_yaw_t=t))
# result: None (not ready / no new frame) or
# {t, velocity: [vx, vy], pose: [x, y, yaw], stationary, ready, latency_us,
#  velocity_cov,   # [vx, vy] 2x2, row-major
#  pose_cov,       # [x, y, yaw] 3x3, row-major, matches the returned pose, maps directly to a ROS covariance
#  uncertainty,    # {'aleatoric_std', 'epistemic_std', 'epistemic_speed_std'}
#  quality}        # 'ok' | 'warn' | 'alarm': 1 s smoothed seed disagreement vs validation P99 / P99.9
```

## Input conventions

- Timestamps are monotonically increasing seconds. IMU units are rad/s and m/s²; the specific force while standing
  upright is about `[0, 0, +9.81]`. Sensor axes: x forward, y left, z up.
- `q` / `dq` are the eight hip and knee joints in rad and rad/s; the order is `g1_odom.common.JOINTS` (left then right
  leg: hip_pitch, hip_roll, hip_yaw, knee).
- Sensors run at 500 Hz, network features at 50 Hz. Normalization is part of the ONNX graph; feature names, rates,
  window, label frame and calibration parameters are stored in the model metadata.
- The simulated IMU sits at the pelvis origin with the pelvis attitude. For a real IMU apply the extrinsic transform
  and handle lever-arm effects if the IMU is offset.
- Velocity is in the gravity-aligned pelvis heading frame. Pose is relative to the moment the estimator became ready;
  yaw is zero at that moment and wraps continuously across ±π.
- A missing or invalid `imu_yaw` raises an error; there is no gyro-integration fallback. `imu_yaw_t` must not be later
  than the sample and not more than 20 ms older (a rejection limit, not a recommended latency).
- The network's gravity feature is built from the raw gyro and accelerometer and needs at least 0.75 s of continuous
  standing still for calibration; the simulation provides 1 s on a fixed stand at the start.

## Uncertainty and limits

- `yaw_drift_deg_min` defaults to 1°/min; replace it with the value measured for your IMU.
- The covariance is only calibrated on the training distribution. When `quality` is warn/alarm, the fusion side should
  down-weight the odometry or use it only as an alert.
- Stop or reset the odometry after a fall.

## Release files

In `releases/`:

- `estimator.onnx`: the three-seed ensemble. Input `features` (1 × 25 × 25), outputs `velocity`, `aleatoric_var`,
  `epistemic_var`; the `calibration` metadata carries the calibration parameters.
- `members/estimator_seed{0,1,2}.onnx`: single seeds with outputs `velocity`, `log_var`.
- `calibration.json`, `model.json` (validation evidence for window and mean source; label / trainer / training data
  hashes), `runtime.json` (output field conventions).
- `implementation.json`: hash snapshot of the implementation files at release time. Its paths use the layout from
  before the directory cleanup (`g1_odom/…`, `web/…`) and are kept as a historical record only.

## Web console

```bash
scripts/run_web.sh               # http://127.0.0.1:18078, optionally --host 0.0.0.0 --port 18078
```

Two pages:

- "Odometry": the page background is the live G1 flat-ground simulation (MuJoCo offscreen rendering, 1280×720, 15 fps,
  pushed as an MJPEG stream on `/api/stream`). Velocity command, trajectory, velocity plots, horizontal error with 2σ,
  position 1σ and estimate quality float over it as translucent panels. The button at the top right, or the `H` key,
  hides and shows the panels.
- "Training and report": data, pipeline stages and release status; start or stop the pipeline, and see accuracy,
  uncertainty coverage and online regression results.

Flat-ground walking uses the official Unitree G1 velocity policy (`assets/unitree_g1/policy.onnx`, loaded in plain
MuJoCo, no mjlab). It only drives the simulation and is independent of the odometry network. The server listens on
localhost by default and has no authentication; use SSH port forwarding or a listen address for remote access.

## ROS 2 node

`ros2/legged_neural_odometry` is a ROS 2 Humble node. It subscribes to `/lowstate` (`unitree_hg/msg/LowState`, reliable,
keep-last 10), takes the G1 SDK joint indices `0, 1, 2, 3, 6, 7, 8, 9` (hip_pitch, hip_roll, hip_yaw, knee of the left
and right leg) and the gyro, accelerometer and `rpy[2]` of `imu_state`. Once features are ready it publishes
`/g1/odom` (`nav_msgs/Odometry`) at about 50 Hz. `frame_id` defaults to `odom` and `child_frame_id` to `pelvis`. The pose
is relative to the first ready moment. `twist.linear.x/y` is the heading-frame velocity and `angular.z` is the gyro z of
the frame that closes it. The reset service is `/legged_neural_odometry/reset`.

`LowState` has no timestamp, so the node uses its own clock. A gap over 100 ms, time going backwards or a non-finite
value resets the estimator. Nothing is published before the standstill calibration completes or while the window is not
full. The accelerometer should read about 9.81 m/s² at rest; if nothing is ever published, the acceleration norm in the
log tells you whether the driver sends accelerations in g.

The `unitree_hg` message definitions are in `ros2/unitree_hg` and contain only `LowState`, `IMUState` and `MotorState`
with fields identical to the Unitree `/lowstate`. Building, packaging and running do not depend on
`uqi-r11-robot-interface`. When Unitree changes the `LowState` fields, update this copy too, otherwise deserialization
is misaligned.

On a development machine you only need to source ROS 2 Humble:

```bash
colcon build --base-paths ros2 --packages-select unitree_hg legged_neural_odometry
source install/setup.bash
ros2 launch legged_neural_odometry legged_neural_odometry.launch.py
```

Packaging, installing and removing. Prebuilt `amd64` and `arm64` debs are attached to every GitHub Release by
`.github/workflows/release.yml`; to build locally:

```bash
bash scripts/build-deb.sh        # writes to dist/
sudo dpkg -i dist/legged-neural-odometry_1.0.0_$(dpkg --print-architecture).deb
sudo dpkg -r legged-neural-odometry
```

The deb installs into `/opt/legged-neural-odometry`; the package and service are both named
`legged-neural-odometry`. The service starts after installation and keeps running while `/lowstate` has no data.
Removal stops the service and disables it at boot. The version comes from `pyproject.toml` (currently 1.0.0); the
build checks that it matches `Cargo.toml` and `package.xml`.
