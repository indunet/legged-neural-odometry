"""Planar stationary detector."""
import numpy as np
from .common import FEATURE_NAMES, JOINTS
WINDOW=25
FRAME_RATE=50.
LABEL_TAU=.2
NO_ANKLE_JOINTS=JOINTS

class StandstillDetector:
    """Standstill degree s in [0, 1] from the feature frames alone (no command, no truth).

    A frame is quiet when, over the last ``window`` frames, every hip/knee joint
    speed, the gyro norm and the spread of the specific-force norm stay below
    their thresholds (set from the full_02 data: ~80% of truly static frames, <0.3%
    of moving ones). Ankles are not used. s rises towards 1 with time constant
    ``tau`` while quiet and drops to 0 on the first non-quiet frame, so a stop is
    not cut short and a restart is not delayed.

    :meth:`gate` scales (vx, vy) by ``1 - s * w``, with w = 1 below ``speed_full`` and
    0 above ``speed_off`` (estimated speed): after a stop the pelvis can still settle
    at 1-5 cm/s with quiet hips and knees, which the network tracks correctly.
    """

    def __init__(self, window=WINDOW, dq_max=0.12, gyro_max=0.06, accel_std_max=0.10, tau=LABEL_TAU,
                 speed_full=0.01, speed_off=0.02):
        self.window, self.tau = window, tau
        self.dq_max, self.gyro_max, self.accel_std_max = dq_max, gyro_max, accel_std_max
        self.speed_full, self.speed_off = speed_full, speed_off
        self.dq_index = np.array([FEATURE_NAMES.index(f"dq_{j}") for j in NO_ANKLE_JOINTS])
        self.reset()

    def reset(self):
        self._history = []
        self.value = 0.

    def update(self, frame):
        frame = np.asarray(frame, dtype=float)
        self._history.append((np.abs(frame[self.dq_index]).max(), np.linalg.norm(frame[0:3]),
                              np.linalg.norm(frame[3:6])))
        del self._history[:-self.window]
        dq, gyro, accel = np.array(self._history).T
        quiet = (len(self._history) == self.window and dq.max() < self.dq_max
                 and gyro.max() < self.gyro_max and accel.std() < self.accel_std_max)
        if quiet:
            self.value += (1. - np.exp(-1. / FRAME_RATE / self.tau)) * (1. - self.value)
        else:
            self.value = 0.
        return self.value

    def gate(self, vx, vy, s=None):
        """Gated (vx, vy); scalars use the current s, arrays need the matching s array."""
        s = self.value if s is None else s
        w = np.clip((self.speed_off - np.hypot(vx, vy)) / (self.speed_off - self.speed_full), 0., 1.)
        keep = 1. - s * w
        return vx * keep, vy * keep

    def gate_velocity(self, velocity, s=None):
        """Gate the full translation vector, including height, while sensors are quiet.

        Use total speed so real vertical settling is preserved even when vx/vy
        are small. The same speed taper as the planar gate avoids a hard cutoff.
        Angular velocity is deliberately excluded.
        """
        velocity = np.asarray(velocity, dtype=float)
        if velocity.ndim < 1 or velocity.shape[-1] not in (2, 3):
            raise ValueError("expected (..., 2) or (..., 3) translational velocity")
        s = self.value if s is None else np.asarray(s)
        speed = np.linalg.norm(velocity, axis=-1)
        weight = np.clip((self.speed_off - speed) / (self.speed_off - self.speed_full), 0., 1.)
        return velocity * np.expand_dims(1. - s * weight, -1)

    def episode(self, features):
        self.reset()
        return np.array([self.update(f) for f in features])

