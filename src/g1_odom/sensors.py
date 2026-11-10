"""Sensor-only attitude, calibration and feature construction. No simulator import."""
from dataclasses import dataclass
from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class SensorSample:
    t: float
    gyro: np.ndarray
    accel: np.ndarray
    q: np.ndarray
    dq: np.ndarray
    # Hardware: Unitree IMU rpy[2], transformed to the pelvis frame, in radians.
    # This is a separate heading input; it is not a neural-network feature.
    imu_yaw: float | None = None
    imu_yaw_t: float | None = None

class Attitude:
    """Complementary attitude with stationary gyro calibration; no truth argument."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.R = np.eye(3)
        self.bias = None
        self.t = None
        self.calibration = deque(maxlen=400)
        self.calibration_start = None
        self.last_calibration_t = None
        self.accel_lp = None
        self.ready = False
        self.yaw = 0.

    def update(self, sample):
        g, a = np.asarray(sample.gyro), np.asarray(sample.accel)
        if not np.isfinite(np.r_[g, a, sample.q, sample.dq, sample.t]).all():
            raise ValueError('non-finite sensor sample')
        dt = 0. if self.t is None else sample.t - self.t
        if dt < 0 or dt > .1:
            raise ValueError('sensor timestamps must increase with gaps <= 100 ms')
        if self.t is not None and dt == 0:
            raise ValueError('duplicate sensor timestamp')
        self.t = float(sample.t)
        if self.bias is None:
            quiet = np.max(np.abs(sample.dq)) < .08 and np.linalg.norm(g) < .08 and abs(np.linalg.norm(a)-9.81) < .65
            if not quiet:
                self.calibration.clear()
                self.calibration_start = None
                return np.array([0., 0., -1.])
            if self.calibration_start is None:
                self.calibration_start = sample.t
            self.calibration.append(np.r_[g, a])
            if sample.t - self.calibration_start >= .75 and len(self.calibration) >= 50:
                rows = np.array(self.calibration)
                if np.max(rows[:, :3].std(0)) < .02 and np.max(rows[:, 3:].std(0)) < .25:
                    self.bias = rows[:, :3].mean(0)
                    up = rows[:, 3:].mean(0)
                    up /= np.linalg.norm(up)
                    # R maps measured body-up onto world-up, with arbitrary initial yaw.
                    self.R = Rotation.align_vectors([[0., 0., 1.]], [up])[0].as_matrix()
                    self.ready = True
            return self.R.T @ np.array([0., 0., -1.])
        omega = g - self.bias
        self.R = self.R @ Rotation.from_rotvec(omega * dt).as_matrix()
        self.accel_lp = a.copy() if self.accel_lp is None else self.accel_lp + dt/(.35+dt)*(a-self.accel_lp)
        # Downweight dynamic acceleration rather than interpreting it as tilt.
        confidence = np.clip(1.-abs(np.linalg.norm(self.accel_lp)-9.81)/1.5, 0., 1.)
        measured_up = self.accel_lp / max(np.linalg.norm(self.accel_lp), 1e-9)
        predicted_up = self.R.T @ np.array([0., 0., 1.])
        correction = .35 * confidence * np.cross(measured_up, predicted_up)
        self.R = self.R @ Rotation.from_rotvec(correction * dt).as_matrix()
        self.yaw = float(np.arctan2(self.R[1, 0], self.R[0, 0]))
        return self.R.T @ np.array([0., 0., -1.])

class FeatureBuilder:
    def __init__(self):
        self.reset()

    def reset(self):
        self.attitude = Attitude()
        self.rows = []
        self.last_frame_t = None

    def push(self, sample):
        if np.shape(sample.q) != (8,) or np.shape(sample.dq) != (8,):
            raise ValueError('expected eight named hip/knee positions and velocities')
        gravity = self.attitude.update(sample)
        self.rows.append(np.r_[sample.gyro, sample.accel])
        if self.last_frame_t is None:
            self.last_frame_t = sample.t
        if sample.t - self.last_frame_t < .02 - 1e-8:
            return None
        self.last_frame_t = sample.t
        mean = np.mean(self.rows, axis=0)
        self.rows.clear()
        return np.r_[mean, gravity, sample.q, sample.dq].astype(np.float32)
