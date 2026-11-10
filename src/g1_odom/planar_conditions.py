"""Operating-condition matrix for data generation.

Command schedules (continuous walking and stop/go) and the command/execution
mismatch. Physics, sensors and disturbances are
sampled per G1 episode in ``dataset.configs``.
"""
import numpy as np

COMMAND_LOW = np.array([-0.5, -0.5, -0.8])
COMMAND_HIGH = np.array([1.2, 0.5, 0.8])

MODES = ("stand", "step_in_place", "slow_forward", "mid_forward", "fast_forward",
         "backward", "lateral", "turn_in_place", "arc", "diagonal", "random")
MODE_WEIGHTS = np.array([1.2, 0.5, 1.2, 1.2, 1., 0.8, 0.8, 0.8, 1.2, 0.8, 0.8])


def u(rng, low, high):
    return float(rng.uniform(low, high))


def mode_target(rng, mode):
    sign = lambda: float(rng.choice([-1., 1.]))
    if mode == "stand":
        return [0., 0., 0.]
    if mode == "step_in_place":
        return [u(rng, -0.05, 0.05), u(rng, -0.03, 0.03), u(rng, -0.05, 0.05)]
    if mode == "slow_forward":
        return [u(rng, 0.03, 0.3), 0., 0.]
    if mode == "mid_forward":
        return [u(rng, 0.3, 0.7), 0., 0.]
    if mode == "fast_forward":
        return [u(rng, 0.7, 1.2), 0., 0.]
    if mode == "backward":
        return [u(rng, -0.5, -0.08), 0., 0.]
    if mode == "lateral":
        return [0., sign() * u(rng, 0.08, 0.5), 0.]
    if mode == "turn_in_place":
        return [0., 0., sign() * u(rng, 0.15, 0.8)]
    if mode == "arc":
        return [u(rng, 0.15, 0.9), 0., sign() * u(rng, 0.1, 0.6)]
    if mode == "diagonal":
        return [u(rng, -0.3, 0.8), sign() * u(rng, 0.08, 0.4), 0.]
    if mode == "random":
        return rng.uniform(COMMAND_LOW, COMMAND_HIGH).tolist()
    raise ValueError(mode)


def sample_schedule(rng, duration, modes=None):
    """Segments of 1-6 s; the first 1-2 s always stand so the window fills at rest."""
    segments = [{"t0": 0., "t1": u(rng, 1., 2.), "mode": "stand", "target": [0., 0., 0.],
                 "transition": "step", "ramp": 0.}]
    weights = MODE_WEIGHTS / MODE_WEIGHTS.sum()
    while segments[-1]["t1"] < duration:
        t0 = segments[-1]["t1"]
        if modes is not None:
            mode = str(rng.choice(modes))
        else:
            mode = str(rng.choice(MODES, p=weights))
        ramp = rng.random() < 0.5
        segments.append({"t0": t0, "t1": min(duration, t0 + u(rng, 1., 6.)), "mode": mode,
                         "target": mode_target(rng, mode),
                         "transition": "ramp" if ramp else "step",
                         "ramp": u(rng, 0.5, 2.) if ramp else 0.})
    return segments


STOP_WALK_MODES = ("slow_forward", "mid_forward", "fast_forward", "backward", "lateral", "arc", "diagonal")


def stop_schedule(rng, duration):
    """Walk 3-8 s, then stand 5-15 s (step or ramp stop); 1-2 repeats within ``duration``.

    The first 1-2 s stand so the window fills at rest. Fills the duration with standing at the end.
    """
    segments = [{"t0": 0., "t1": u(rng, 1., 2.), "mode": "stand", "target": [0., 0., 0.],
                 "transition": "step", "ramp": 0.}]
    for _ in range(int(rng.integers(1, 3))):
        t0 = segments[-1]["t1"]
        if t0 >= duration - 8.:
            break
        mode = str(rng.choice(STOP_WALK_MODES))
        t1 = min(duration, t0 + u(rng, 3., 8.))
        ramp = rng.random() < 0.5
        segments.append({"t0": t0, "t1": t1, "mode": mode, "target": mode_target(rng, mode),
                         "transition": "ramp" if ramp else "step", "ramp": u(rng, 0.5, 2.) if ramp else 0.})
        t2 = min(duration, t1 + u(rng, 5., 15.))
        # the stop itself: a ramped or immediate return to zero command
        soft = rng.random() < 0.5
        segments.append({"t0": t1, "t1": t2, "mode": "stand", "target": [0., 0., 0.],
                         "transition": "ramp" if soft else "step", "ramp": u(rng, 0.3, 1.) if soft else 0.})
    if segments[-1]["t1"] < duration:
        segments[-1]["t1"] = duration
    return segments


def nominal_mismatch():
    return {"scale": [1., 1., 1.], "bias": [0., 0., 0.], "drift_amp": 0., "drift_period": 10.,
            "drift_phase": 0., "jitter_std": [0., 0., 0.], "jitter_tau": 0.5}


def sample_mismatch(rng):
    m = nominal_mismatch()
    if rng.random() < 0.8:
        m["scale"] = rng.uniform(0.5, 1.5, 3).tolist()
    if rng.random() < 0.5:
        m["bias"] = [u(rng, -0.05, 0.05), u(rng, -0.05, 0.05), u(rng, -0.08, 0.08)]
    if rng.random() < 0.5:
        m.update(drift_amp=u(rng, 0., 0.15), drift_period=u(rng, 5., 20.),
                 drift_phase=u(rng, 0., 2 * np.pi))
    if rng.random() < 0.3:
        m.update(jitter_std=[0.05, 0.03, 0.08], jitter_tau=u(rng, 0.2, 1.))
    return m


class CommandSchedule:
    """Requested command from the schedule, and the command actually given to the policy."""

    def __init__(self, config, rng):
        self.segments = config["schedule"]
        self.m = config["mismatch"]
        self.rng = rng
        self.jitter = np.zeros(3)
        self._index = 0

    def segment(self, t):
        while self._index + 1 < len(self.segments) and t >= self.segments[self._index]["t1"]:
            self._index += 1
        return self._index, self.segments[self._index]

    def request(self, t, dt):
        index, s = self.segment(t)
        target = np.array(s["target"], dtype=float)
        if s["transition"] == "ramp" and index > 0 and t - s["t0"] < s["ramp"]:
            previous = np.array(self.segments[index - 1]["target"], dtype=float)
            target = previous + (target - previous) * (t - s["t0"]) / s["ramp"]
        std = np.array(self.m["jitter_std"])
        if std.any():
            a = dt / self.m["jitter_tau"]
            self.jitter += -a * self.jitter + std * np.sqrt(2 * a) * self.rng.standard_normal(3)
        return np.clip(target + self.jitter, COMMAND_LOW, COMMAND_HIGH), index

    def policy_command(self, request, t):
        m = self.m
        drift = m["drift_amp"] * np.sin(2 * np.pi * t / m["drift_period"] + m["drift_phase"])
        command = request * (np.array(m["scale"]) + drift) + np.array(m["bias"]) * (np.abs(request) > 1e-6)
        return np.clip(command, COMMAND_LOW, COMMAND_HIGH)


