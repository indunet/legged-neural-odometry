import numpy as np
import pytest
import torch

from g1_odom.common import FRAME_DT, ROOT
from g1_odom.evaluate import integrate, relative_pose
from g1_odom.labels import lag_correction, lag_gain, lowpass, world_lowpass_labels
from g1_odom.model import TCN, TCNGaussian
from g1_odom.runtime import Odometry
from g1_odom.sensors import SensorSample
from g1_odom.uncertainty import PoseCovariance, velocity_variance

RELEASE = ROOT/'releases'
CAL = {'aleatoric_scale': 1., 'epistemic_scale': 2., 'static_var': [1e-6, 1e-6], 'var_floor': 1e-8,
       'bias_fraction': .3, 'correlation_time_s': .15, 'bias_time_s': None, 'pose_scale': 1.2,
       'quality_warn': .01, 'quality_alarm': .03}


def swaying_walk(n=3000, seed=0):
    """Curved walk with per-step pelvis yaw and lateral sway, sampled at 50 Hz."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)*FRAME_DT
    speed = .4+.2*np.sin(.05*t)+.02*rng.normal(size=n).cumsum()/np.sqrt(n)
    heading = .3*np.sin(.07*t)+.04*np.sin(2*np.pi*t/.6)
    path = np.cumsum(np.c_[speed*np.cos(heading), speed*np.sin(heading)], axis=0)*FRAME_DT
    sway = .03*np.sin(2*np.pi*t/1.2)
    xy = path+np.c_[-np.sin(heading)*sway, np.cos(heading)*sway]
    return t, np.c_[xy, np.zeros(n), heading]


def test_perfect_labels_reconstruct_position_below_1cm():
    t, pose = swaying_walk()
    label = world_lowpass_labels(pose, t)
    ends = np.arange(75, len(t))
    yaw = pose[ends, 3]
    estimate = integrate(label[ends, :2], yaw)
    estimate[:, :2] += lag_correction(label[ends], yaw-yaw[0], lag_gain())
    error = np.linalg.norm(estimate[:, :2]-relative_pose(pose[ends])[:, :2], axis=1)
    assert error.max() < .01
    plain = integrate(label[ends, :2], yaw)
    assert np.linalg.norm(plain[-1, :2]-relative_pose(pose[ends])[-1, :2]) > 10*error.max()


@pytest.mark.skipif(not (ROOT/'data/g1_planar/stop_val').exists(), reason='recorded G1 data absent')
def test_perfect_labels_on_recorded_g1_validation():
    from g1_odom.train import episode_loader
    for row in episode_loader(ROOT/'data/g1_planar/stop_val')[:10]:
        ends = np.arange(75, len(row['time']))
        yaw = np.unwrap(row['pose'][ends, 3])
        estimate = integrate(row['label'][ends, :2], yaw)
        estimate[:, :2] += lag_correction(row['label'][ends], yaw-yaw[0], lag_gain())
        truth = relative_pose(np.c_[row['pose'][ends, :3], yaw])
        assert np.linalg.norm(estimate[:, :2]-truth[:, :2], axis=1).max() < .01


def test_lag_compensation_identity_is_exact():
    rng = np.random.default_rng(3)
    v = rng.normal(size=(400, 2))
    f = lowpass(v)
    trapezoid = (.5*(f[1:]+f[:-1])).sum(0)*FRAME_DT
    np.testing.assert_allclose(trapezoid+lag_gain()*(f[-1]-f[0]), v[1:].sum(0)*FRAME_DT, atol=1e-12)


def test_variance_training_keeps_mean_bit_identical():
    torch.manual_seed(0)
    mean, std = np.zeros(25), np.ones(25)
    base = TCN(25, mean, std, n_out=2)
    model = TCNGaussian(25, mean, std, n_out=2)
    missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and all(k.startswith('var_head.') for k in missing)
    model.freeze_mean()
    x, y = torch.randn(64, 25, 25), torch.randn(64, 2)*.1
    before = base(x).detach()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    for _ in range(5):
        velocity, log_var = model(x)
        loss = (.5*(log_var+(y-velocity)**2*torch.exp(-log_var))).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    velocity, log_var = model(x)
    assert torch.equal(velocity, before)
    assert torch.isfinite(log_var).all()


def test_pose_covariance_is_psd_and_grows_while_walking_one_way():
    # A heading-frame bias cancels when the robot turns back, so growth is only
    # guaranteed for one-way walking; PSD must hold for any heading sequence.
    rng = np.random.default_rng(5)
    tracker = PoseCovariance(CAL, yaw_drift_deg_min=1.)
    for _ in range(3000):
        std = np.sqrt(velocity_variance(rng.uniform(1e-6, 1e-4, 2), rng.uniform(0, 1e-5, 2), rng.uniform(), CAL))
        P = tracker.step(FRAME_DT, rng.uniform(-np.pi, np.pi), rng.normal(size=2)*.4, std)
        assert np.linalg.eigvalsh(P).min() > -1e-14
    tracker, previous = PoseCovariance(CAL, yaw_drift_deg_min=1.), 0.
    for _ in range(3000):
        std = np.sqrt(velocity_variance(rng.uniform(1e-6, 1e-4, 2), rng.uniform(0, 1e-5, 2), rng.uniform(), CAL))
        P = tracker.step(FRAME_DT, .7, [rng.uniform(.1, .6), 0.], std)
        trace = np.trace(P[:3, :3])
        assert trace >= previous-1e-15
        previous = trace


def test_velocity_scale_fit_recovers_known_scales():
    from g1_odom.train import fit_velocity_scales
    rng = np.random.default_rng(7)
    a, e = rng.uniform(1e-5, 1e-4, (40000, 2)), rng.uniform(1e-6, 4e-5, (40000, 2))
    residual = rng.normal(size=(40000, 2))*np.sqrt(1.5**2*a+3**2*e)
    fit = fit_velocity_scales(residual, a, e, np.ones(40000))
    assert abs(fit['aleatoric_scale']-1.5) < .15 and abs(fit['epistemic_scale']-3) < .4


def sample(t, yaw=0.):
    s = SensorSample(t, np.zeros(3), np.array([0., 0., 9.81]), np.zeros(8), np.zeros(8), yaw, None)
    if t > .9:
        s.q[:] = .1*np.sin(t*3.)
        s.dq[:] = .3*np.cos(t*3.)
    return s


@pytest.mark.skipif(not (RELEASE/'estimator.onnx').exists(), reason='release not built')
def test_runtime_outputs_covariance_and_matches_member_ensemble():
    import json
    odom = Odometry()
    model = json.loads((RELEASE/'model.json').read_text())
    assert odom.window == model['window'] and odom.lag_gain == pytest.approx(lag_gain())
    members = Odometry(sorted((RELEASE/'members').glob('estimator_seed*.onnx')),
                       calibration=RELEASE/'calibration.json')
    samples = [sample(i*.002, .2) for i in range(2500)]
    rows = [(a, b) for a, b in ((odom.update(s), members.update(s)) for s in samples) if a is not None]
    assert len(rows) > 10
    previous = 0.
    for a, b in rows:
        P = np.array(a['pose_cov']).reshape(3, 3)
        assert np.linalg.eigvalsh(P).min() > -1e-12 and np.trace(P) >= previous-1e-15
        previous = np.trace(P)
        assert np.all(np.array(a['velocity_cov']).reshape(2, 2).diagonal() > 0)
        assert a['quality'] in ('ok', 'warn', 'alarm')
        if model['mean_source'] == 'ensemble':
            np.testing.assert_allclose(a['velocity'], b['velocity'], atol=1e-6)
            np.testing.assert_allclose(a['pose_cov'], b['pose_cov'], rtol=1e-3, atol=1e-12)
