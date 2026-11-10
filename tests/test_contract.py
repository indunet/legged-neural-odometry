import inspect

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from g1_odom.common import FEATURE_NAMES,JOINTS,ROOT
from g1_odom.sensors import SensorSample,FeatureBuilder,Attitude
from g1_odom.evaluate import integrate,relative_pose
from g1_odom.dataset import configs
from g1_odom.simulation import Simulation

def sample(t,gyro=None,accel=None):
    return SensorSample(t,np.array([0.,0.,0.] if gyro is None else gyro),
                        np.array([0.,0.,9.81] if accel is None else accel),np.zeros(8),np.zeros(8))

def test_public_sensor_contract():
    assert len(FEATURE_NAMES)==25 and len(JOINTS)==8
    assert set(SensorSample.__dataclass_fields__)=={'t','gyro','accel','q','dq','imu_yaw','imu_yaw_t'}
    text=inspect.getsource(__import__('g1_odom.runtime',fromlist=['Odometry']))
    assert 'mujoco' not in text and 'torch' not in text
    assert all('ankle' not in n and 'command' not in n and 'action' not in n for n in FEATURE_NAMES)

def test_sensor_only_bias_calibration():
    attitude=Attitude();bias=np.array([.004,-.003,.005])
    for i in range(450):attitude.update(sample((i+1)*.002,gyro=bias))
    assert attitude.ready
    np.testing.assert_allclose(attitude.bias,bias,atol=1e-12)
    for i in range(450,950):attitude.update(sample((i+1)*.002,gyro=bias+[0,0,.2]))
    assert abs(attitude.yaw-.2)<.002

def test_motion_blocks_calibration():
    attitude=Attitude()
    for i in range(600):attitude.update(sample((i+1)*.002,gyro=[0,0,.3]))
    assert not attitude.ready

def test_tilt_gravity_and_stationary_propagation():
    attitude=Attitude();r=Rotation.from_euler('xy',[.2,-.15]).as_matrix()
    a=r.T@np.array([0,0,9.81])
    for i in range(1000):g=attitude.update(sample((i+1)*.002,accel=a))
    np.testing.assert_allclose(g,r.T@[0,0,-1],atol=1e-5)

def test_invalid_timestamps():
    b=FeatureBuilder();b.push(sample(.1))
    with pytest.raises(ValueError):b.push(sample(.1))
    with pytest.raises(ValueError):b.push(sample(.05))

def test_frame_rate_and_reset():
    b=FeatureBuilder();frames=[]
    for i in range(501):
        f=b.push(sample(i*.002))
        if f is not None:frames.append(f)
    assert len(frames)==50 and frames[0].shape==(25,) and b.attitude.ready
    b.reset();assert not b.attitude.ready

def test_integral_and_relative_frame():
    v=np.tile([1.,0.,.2],(501,1));yaw=np.full(501,np.pi/2)
    p=integrate(v,yaw)
    np.testing.assert_allclose(p[-1,:3],[10.,0.,2.],atol=1e-10)
    q=p.copy();q[:,:2]=np.c_[-p[:,1],p[:,0]]+[3,5];q[:,3]+=np.pi/2;q[:,2]+=7
    np.testing.assert_allclose(relative_pose(q),p,atol=1e-10)

def test_split_isolation_and_exact_counts():
    jobs=configs();assert len(jobs)==3110
    seeds=[j['seed'] for j in jobs];assert len(seeds)==len(set(seeds))
    assert {j['course']['kind'] for j in jobs}=={'flat'}

def test_native_g1_joint_mapping_and_real_fixture():
    from g1_odom.simulation import Simulation
    s=Simulation(noise=False);b=FeatureBuilder()
    assert s.model.nu==29 and len(set(s.policy.qids))==29
    for i in range(450):s.step();b.push(s.observe())
    assert b.attitude.ready
    np.testing.assert_allclose(s.data.sensor('odom_accel').data,[0,0,9.81],atol=.01)
    s.step();assert s.data.time>.9

def test_model_copy_has_no_source_runtime_dependency():
    for name in ('common.py','sensors.py','simulation.py','runtime.py','model.py','train.py'):
        text=(ROOT/'src/g1_odom'/name).read_text()
        assert 'sys.path.insert' not in text

def test_rpe_aggregation_weights_segments():
    from g1_odom.evaluate import aggregate
    rows=[{'valid':True,'velocity_rmse':[.1,.2],'rpe':{'5':{'segments':1,'position_mean_m':1.,'yaw_mean_deg':2.}}},
          {'valid':True,'velocity_rmse':[.1,.2],'rpe':{'5':{'segments':9,'position_mean_m':0.,'yaw_mean_deg':0.}}}]
    result=aggregate(rows)
    assert result['rpe']['5']['segments']==10
    assert result['rpe']['5']['position_mean_m']==.1

def test_policy_stands_still_at_zero_command():
    """Unitree's deployment zeroes the gait phase for |command| < 0.1; without it the robot marches and drifts."""
    def drift(gate):
        s=Simulation(noise=False,gate_standing=gate)
        while s.data.time<2.:s.step()
        start=s.data.xpos[s.base][:2].copy()
        while s.data.time<6.:s.step()
        return float(np.linalg.norm(s.data.xpos[s.base][:2]-start)),s.fallen()
    gated,fell=drift(True)
    assert not fell and gated<.02

def test_heading_hold_only_steers_while_walking_straight():
    from g1_odom.live import HeadingHold
    h=HeadingHold()
    np.testing.assert_allclose(h([0,0,0],.3),[0,0,0])              # standing: untouched, stays under the stand-still gate
    assert h([.5,0,0],.3)[2]==0                                      # locks heading 0.3 rad
    assert h([.5,0,0],.2)[2]>0 and h([.5,0,0],.4)[2]<0               # steers back toward it
    np.testing.assert_allclose(h([.5,0,.4],1.),[.5,0,.4])          # explicit turn command wins and releases the lock
    assert h([.5,0,0],1.)[2]==0                                      # new heading locked at release
    assert abs(h([.5,0,0],-3.1)[2])<=.4+1e-12                        # bounded correction
