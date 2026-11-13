**English** | [简体中文](zh-CN/method.md)

# Method

- **Data**: 3110 fresh G1 trajectories (seeds from 400000000). The training pool is 2000 general walking plus 600
  stop-and-go trajectories, split 90/10 by whole trajectory. There are 200 + 60 extra validation trajectories and test
  sets of 100 IID, 60 OOD, 30 long trajectories and 60 stop-and-go. Commands follow continuous-walking and stop-and-go
  schedules, with randomized friction, mass, payload, PD gains, torque limits, sensor noise, pushes and yaw
  disturbances on the G1 model.
- **Labels**: differences of the recorded pelvis position, low-passed in the world frame with a first-order filter
  (0.2 s), then rotated into the current heading. The integrator adds `(τ + dt/2)·(v_now − v_start)` (world frame) to
  cancel the filter lag exactly; integrating the labels as network outputs gives a position error below 0.02 cm.
- **Mean network**: TCN24 (about 6k parameters), input 25 frames × 25 features, Huber(0.1), AdamW (lr 0.003),
  OneCycleLR, batch size 2048, 40 × 600 steps, model selected by validation Huber loss; seeds 0/1/2 are trained from
  scratch. A 50-frame window was worse on validation data and is not used.
- **Uncertainty**: with the mean frozen, each seed trains a variance head (Gaussian negative log-likelihood), which
  gives the aleatoric uncertainty; the disagreement between the three seeds gives the epistemic uncertainty. The three
  seeds are merged into one ONNX graph. Both variances are scaled separately on the extra validation set and then
  propagated into the pose covariance through the error state `[px, py, yaw, bx, by]`: `b` is a slowly varying bias in
  units of the velocity standard deviation, the remainder acts as white noise given its correlation time, and yaw is a
  random walk whose 1σ after 60 s defaults to 1°.
- **Simulation policy**: flat-ground walking uses the official Unitree G1 velocity policy
  (`assets/unitree_g1/policy.onnx`, run in plain MuJoCo). As in the official deployment, the gait phase input is zeroed
  when the command norm is below 0.1 so the robot stands still; otherwise the policy steps in place and drifts slowly
  (an early version missed this and was fixed, with data, training and evaluation redone).
- **Selection rule**: window, mean source and all calibration parameters use only the extra validation set; the test
  sets are evaluated once after locking.

Results are in [reports/report.md](../reports/report.md) (Chinese).
