"""Write a short SensorSample trace and the Python Odometry outputs."""
import json
import math
from pathlib import Path

import numpy as np

from g1_odom.runtime import Odometry
from g1_odom.sensors import SensorSample

HERE = Path(__file__).resolve().parent


def tilt():
    raw = np.array([0.35, -0.22, 9.7])
    return raw * (9.81 / np.linalg.norm(raw))


ACCEL_STILL = tilt()
PHASE = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7])


def make_sample(i):
    t = i * 0.002
    if t < 1.0:
        gyro = np.array([0.004, -0.003, 0.005])
        accel = ACCEL_STILL
        q = np.zeros(8)
        dq = np.zeros(8)
        yaw = 0.2
    else:
        gyro = np.array([0.02 * math.sin(2.0 * t), -0.015 * math.cos(1.7 * t), 0.12])
        accel = ACCEL_STILL + np.array([0.15 * math.sin(5.0 * t), 0.1 * math.cos(4.0 * t), 0.05 * math.sin(t)])
        q = 0.25 * np.sin(t * PHASE)
        dq = 0.5 * np.cos(t * PHASE)
        yaw = 0.2 + 0.12 * (t - 1.0)
    return SensorSample(t, gyro, accel, q, dq, imu_yaw=yaw, imu_yaw_t=t)


def dump(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def main():
    odom = Odometry()
    samples = []
    results = []
    for i in range(1400):
        sample = make_sample(i)
        samples.append({
            "t": sample.t,
            "gyro": dump(sample.gyro),
            "accel": dump(sample.accel),
            "q": dump(sample.q),
            "dq": dump(sample.dq),
            "imu_yaw": float(sample.imu_yaw),
            "imu_yaw_t": float(sample.imu_yaw_t),
        })
        result = odom.update(sample)
        if result is None:
            results.append(None)
        else:
            results.append({
                "t": result["t"],
                "velocity": result["velocity"],
                "pose": result["pose"],
                "stationary": result["stationary"],
                "stationary_degree": result["stationary_degree"],
                "velocity_cov": result["velocity_cov"],
                "pose_cov": result["pose_cov"],
                "quality": result["quality"],
                "epistemic_speed_std": result["uncertainty"]["epistemic_speed_std"],
            })
    ready = sum(item is not None for item in results)
    path = HERE / "golden.json"
    path.write_text(json.dumps({"samples": samples, "results": results}, allow_nan=False) + "\n")
    print(f"wrote {path} ({ready} ready frames)")


if __name__ == "__main__":
    main()
