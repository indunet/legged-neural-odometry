import numpy as np
import pytest

from g1_odom.runtime import MODEL, Odometry
from g1_odom.sensors import SensorSample

pytestmark = pytest.mark.skipif(not MODEL.exists(), reason='release not built')


def sample(t, yaw=0., yaw_t=None):
    return SensorSample(t, np.zeros(3), np.array([0., 0., 9.81]),
                        np.zeros(8), np.zeros(8), yaw, yaw_t)


def test_external_yaw_required_and_timestamp_checked():
    odom = Odometry()
    for bad in (sample(0., None), sample(0., np.nan), sample(.1, 0., .05),
                sample(.1, 0., .11), sample(.1, 0., np.nan)):
        with pytest.raises(ValueError):
            odom.update(bad)
    assert odom.builder.attitude.t is None  # rejected packets do not advance state


def test_heading_wrap_reset_and_velocity_rotation():
    odom = Odometry()
    odom.lag_gain = 0.
    # A fixed velocity isolates heading/integration from network accuracy.
    zero = np.zeros((1, 2), dtype=np.float32)
    odom.session.run = lambda *_: [np.array([[1., 0.]], dtype=np.float32), zero, zero]
    odom.still.update = lambda _: 0.
    results = []
    for i in range(1501):
        t=i*.002
        yaw=np.deg2rad(170.)+.4*t
        value=odom.update(sample(t, np.arctan2(np.sin(yaw), np.cos(yaw))))
        if value: results.append(value)
    elapsed=results[-1]['t']-results[0]['t']
    np.testing.assert_allclose(results[-1]['pose'][2], .4*elapsed, atol=1e-10)
    np.testing.assert_allclose(results[-1]['pose'][:2],
                               [np.sin(.4*elapsed)/.4, (1-np.cos(.4*elapsed))/.4], atol=2e-5)
    assert abs(odom.builder.attitude.yaw)<1e-10  # external yaw, not internal gyro heading
    odom.reset()
    first=None
    for i in range(501):
        value=odom.update(sample(i*.002, -2.))
        if value and first is None: first=value
    np.testing.assert_array_equal(first['pose'], np.zeros(3))


def test_yaw_does_not_change_network_features_or_velocity():
    a_odom, b_odom = Odometry(), Odometry()
    saw_motion=False
    for i in range(601):
        a, b = sample(i*.002, 1.2), sample(i*.002, -.5)
        for s in (a, b):
            if s.t>.9:
                s.q[:]=.1*np.sin(s.t*3.)
                s.dq[:]=.3*np.cos(s.t*3.)
        x, y = a_odom.update(a), b_odom.update(b)
        if x is not None:
            np.testing.assert_array_equal(x['velocity'], y['velocity'])
            np.testing.assert_array_equal(np.stack(a_odom.frames), np.stack(b_odom.frames))
            saw_motion |= np.linalg.norm(x['velocity'])>1e-6
    assert saw_motion
