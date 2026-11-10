"""Velocity labels rebuilt from the recorded pelvis pose; no simulator import.

Low-passing in the pelvis heading frame, which swings with every step, biases
the integrated position. Instead the recorded world position is differenced,
low-passed in the world frame and only then rotated into the current heading.
For the first-order filter and trapezoidal integration used at runtime,

    true displacement = sum of trapezoids + (tau + dt/2) * (f_now - f_start)

in the world frame, so the 0.2 s filter lag can be removed from the pose.
"""
import numpy as np
from scipy.signal import lfilter

from .common import FRAME_DT

LABEL_TAU = .2
LABEL_FRAME = 'world_lowpass'


def lowpass(x, tau=LABEL_TAU, dt=FRAME_DT):
    """First-order recursion f += dt/(tau+dt) * (x - f), starting from f = 0, per column."""
    alpha = dt/(tau+dt)
    return lfilter([alpha], [1., alpha-1.], np.asarray(x, dtype=float), axis=0)


def lag_gain(tau=LABEL_TAU, dt=FRAME_DT):
    return tau+dt/2


def world_lowpass_labels(pose, time=None, tau=LABEL_TAU, dt=FRAME_DT):
    """Heading-frame [vx, vy, vz] from pose rows [x, y, z, yaw] sampled every ``dt``."""
    pose = np.asarray(pose, dtype=float)
    if pose.ndim != 2 or pose.shape[1] != 4:
        raise ValueError('expected pose rows [x, y, z, yaw]')
    if time is not None and len(time) > 1 and not np.allclose(np.diff(time), dt, atol=2e-5):
        raise ValueError('pose must be sampled at the frame rate')
    world = np.zeros((len(pose), 3))
    world[1:] = np.diff(pose[:, :3], axis=0)/dt
    world = lowpass(world, tau, dt)
    yaw = pose[:, 3]
    c, s = np.cos(yaw), np.sin(yaw)
    return np.c_[c*world[:, 0]+s*world[:, 1], -s*world[:, 0]+c*world[:, 1], world[:, 2]]


def lag_correction(velocity, heading, gain):
    """World-frame displacement to add to an integrated pose, per row.

    ``velocity`` is the heading-frame output (N, >=2) and ``heading`` the yaw
    used for integration, relative to its first row.
    """
    velocity = np.asarray(velocity, dtype=float)
    c, s = np.cos(heading), np.sin(heading)
    world = np.c_[c*velocity[:, 0]-s*velocity[:, 1], s*velocity[:, 0]+c*velocity[:, 1]]
    return gain*(world-world[0])


def relabel(row, tau=LABEL_TAU):
    """Replace a loaded episode's recorded labels; keeps the 3-column layout."""
    row['label'] = world_lowpass_labels(row['pose'], row['time'], tau).astype(np.float32)
    row['label_names'] = np.array(['vx', 'vy', 'vz'])
    row['label_frame'] = LABEL_FRAME
    return row
