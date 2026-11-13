"""G1 flat-ground policy adapter and independent MuJoCo simulation (truth stays here)."""
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import onnxruntime as ort

from .common import ROOT, DT, JOINTS
from .sensors import SensorSample

G1_ROOT = ROOT/'assets/unitree_g1'
FLAT_POLICY = G1_ROOT/'policy.onnx'

def session(path):
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = 1
    opt.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), opt, providers=['CPUExecutionProvider'])

class Policy:
    """Unitree Mjlab G1 velocity policy (flat ground), 50 Hz actions, 500 Hz PD torque."""
    def __init__(self, gate_standing=True):
        # Unitree's deployment code feeds a zero gait phase when |command| < 0.1 so the robot stands still;
        # without it the policy steps in place and drifts. Pass False only to reproduce data recorded before the fix.
        self.gate_standing = gate_standing
        self.path = FLAT_POLICY
        self.session = session(self.path)
        self.meta = self.session.get_modelmeta().custom_metadata_map
        self.names = tuple(self.meta['joint_names'].split(','))
        for attr, key in [('default', 'default_joint_pos'), ('kp', 'joint_stiffness'),
                          ('kd', 'joint_damping'), ('scale', 'action_scale')]:
            val = np.fromstring(self.meta[key], sep=',')
            if val.shape != (29,) or not np.isfinite(val).all():
                raise ValueError(f'invalid policy metadata {key}')
            setattr(self, attr, val)
        self.reset()

    def reset(self):
        self.action = np.zeros(29)
        self.target = self.default.copy()

    def bind(self, model):
        self.qids = np.array([model.joint(n).qposadr[0] for n in self.names])
        self.vids = np.array([model.joint(n).dofadr[0] for n in self.names])
        self.aids = np.array([model.actuator(n).id for n in self.names])
        self.limits = model.actuator_ctrlrange[self.aids].copy()
        self.base = model.body('pelvis').id

    def infer(self, model, data, command):
        R = data.xmat[self.base].reshape(3, 3)
        gyro = data.sensor('odom_gyro').data.copy()
        q, dq = data.qpos[self.qids], data.qvel[self.vids]
        phase = ((data.time - 1.) / .6) % 1.
        gait = np.array([np.sin(2*np.pi*phase), np.cos(2*np.pi*phase)])
        if self.gate_standing and np.linalg.norm(command) < .1:
            gait[:] = 0.
        obs = np.r_[gyro, R.T @ [0., 0., -1.], command, gait, q-self.default, dq, self.action]
        self.action = np.clip(self.session.run(['actions'], {'obs': obs[None].astype(np.float32)})[0][0], -100, 100)
        if not np.isfinite(self.action).all():
            raise FloatingPointError('nonfinite locomotion actions')
        self.target = self.default + self.scale*self.action

    def torque(self, data, kp_scale=1., kd_scale=1.):
        value = self.kp*kp_scale*(self.target-data.qpos[self.qids])-self.kd*kd_scale*data.qvel[self.vids]
        data.ctrl[self.aids] = np.clip(value, self.limits[:, 0], self.limits[:, 1])

def build_model(policy):
    xml_path = G1_ROOT/'g1.xml'
    root = ET.parse(xml_path).getroot()
    compiler = root.find('compiler')
    compiler.set('meshdir', str((xml_path.parent/compiler.get('meshdir', '.')).resolve()))
    world = root.find('worldbody')
    for geom in world.findall('.//geom'):
        geom.set('group', '2' if geom.get('class') == 'visual' else '3')
    motors = ET.SubElement(root, 'actuator')
    for n, kp in zip(policy.names, policy.kp):
        j = root.find(f'.//joint[@name="{n}"]')
        j.set('armature', str(kp/(10*2*np.pi)**2))
        j.set('damping', '0')
        j.set('frictionloss', '.1')
        limit = 139 if ('hip_roll' in n or 'knee' in n) else 88 if ('hip_' in n or n=='waist_yaw_joint') else 50 if ('ankle' in n or 'waist' in n) else 5 if ('wrist_pitch' in n or 'wrist_yaw' in n) else 25
        j.set('actuatorfrcrange', f'{-limit} {limit}')
        ET.SubElement(motors, 'motor', name=n, joint=n, ctrlrange=f'{-limit} {limit}')
    for geom in world.findall('.//geom'):
        if geom.get('class') != 'visual':
            geom.set('condim', '3' if 'foot' in geom.get('name', '') else '1')
            geom.set('friction', '.6 .005 .0001')
    option = root.find('option')
    if option is None:
        option = ET.SubElement(root, 'option')
    option.set('timestep', str(DT))
    option.set('integrator', 'implicitfast')
    base = world.find('.//body[@name="pelvis"]')
    ET.SubElement(world, 'body', name='calibration_fixture', mocap='true', pos='0 0 .85')
    equality = root.find('equality')
    if equality is None:
        equality = ET.SubElement(root, 'equality')
    ET.SubElement(equality,'weld',name='calibration_fixture_weld',body1='pelvis',
                  body2='calibration_fixture',relpose='0 0 0 1 0 0 0',solref='.002 1',
                  solimp='.99999 .99999 .001')
    ET.SubElement(base, 'site', name='odom_imu', pos='0 0 0', size='.005', group='5')
    sensors = root.find('sensor')
    if sensors is None:
        sensors = ET.SubElement(root, 'sensor')
    ET.SubElement(sensors, 'gyro', name='odom_gyro', site='odom_imu')
    ET.SubElement(sensors, 'accelerometer', name='odom_accel', site='odom_imu')
    ET.SubElement(world, 'light', pos='0 0 5', dir='0 0 -1', directional='true')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='100 100 .1', group='0',
                  friction='.8 .005 .0001', rgba='.32 .35 .38 1', contype='1', conaffinity='1')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual,'global',offwidth='960',offheight='640')
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))

class Simulation:
    def __init__(self, seed=0, noise=True, randomize=False, gate_standing=True):
        self.policy = Policy(gate_standing)
        self.model = build_model(self.policy)
        self.policy.bind(self.model)
        self.base = self.model.body('pelvis').id
        self.qids = np.array([self.model.joint(n).qposadr[0] for n in JOINTS])
        self.vids = np.array([self.model.joint(n).dofadr[0] for n in JOINTS])
        self.rng = np.random.default_rng(seed)
        self.noise = noise
        self.gyro_std=.002;self.accel_std=.04
        self.joint_position_std=.0003;self.joint_velocity_std=.008
        self.gyro_walk_std=.00015
        self.randomize = randomize
        self.gyro_bias = self.rng.uniform(-.005,.005,3) if noise else np.zeros(3)
        self.accel_bias = self.rng.uniform(-.06,.06,3) if noise else np.zeros(3)
        self.kp_scale = self.rng.uniform(.9,1.1,29) if randomize else np.ones(29)
        self.kd_scale = self.rng.uniform(.9,1.1,29) if randomize else np.ones(29)
        if randomize:
            self.model.geom_friction[:,0] = self.rng.uniform(.45,1.1)
            scale = self.rng.uniform(.9,1.1)
            self.model.body_mass[1:] *= scale
            self.model.body_inertia[1:] *= scale
        self.reset()

    def reset(self):
        self.data = mujoco.MjData(self.model)
        self.policy.reset()
        self.data.qpos[2] = .85
        self.data.qpos[3:7] = [1,0,0,0]
        self.data.qpos[self.policy.qids] = self.policy.default
        mujoco.mj_forward(self.model,self.data)
        ids = [i for i in range(self.model.ngeom) if 'foot' in (self.model.geom(i).name or '')]
        lowest = min(self.data.geom_xpos[i,2]-self.model.geom_size[i,0]
                     -abs(self.data.geom_xmat[i].reshape(3,3)[2,2])*self.model.geom_size[i,1] for i in ids)
        self.data.qpos[2] -= lowest-.001
        self.initial_qpos = self.data.qpos.copy()
        self.data.mocap_pos[0] = self.data.qpos[:3]
        self.data.mocap_quat[0] = self.data.qpos[3:7]
        self.fixture = self.model.equality('calibration_fixture_weld').id
        self.step_index = 0
        self._gyro_walk = np.zeros(3)
        mujoco.mj_forward(self.model,self.data)

    def observe(self, imu_yaw=False):
        d = self.data
        gyro = d.sensor('odom_gyro').data.copy()
        accel = d.sensor('odom_accel').data.copy()
        q, dq = d.qpos[self.qids].copy(), d.qvel[self.vids].copy()
        if self.noise:
            self._gyro_walk += self.rng.normal(0, self.gyro_walk_std*np.sqrt(DT), 3)
            gyro += self.gyro_bias+self._gyro_walk+self.rng.normal(0,self.gyro_std,3)
            accel += self.accel_bias+self.rng.normal(0,self.accel_std,3)
            q += self.rng.normal(0,self.joint_position_std,8)
            dq += self.rng.normal(0,self.joint_velocity_std,8)
        # Explicit ideal attitude-output emulator. Not measured Unitree hardware
        # accuracy; raw-IMU noise and neural features are unchanged.
        R=d.site_xmat[self.model.site('odom_imu').id].reshape(3,3)
        yaw=float(np.arctan2(R[1,0],R[0,0])) if imu_yaw else None
        return SensorSample(float(d.time), gyro, accel, q, dq, yaw,
                            float(d.time) if imu_yaw else None)

    def truth(self):
        d = self.data
        R = d.xmat[self.base].reshape(3,3)
        yaw = float(np.arctan2(R[1,0],R[0,0]))
        c,s = np.cos(yaw),np.sin(yaw)
        v = d.qvel[:3]
        velocity = np.array([c*v[0]+s*v[1],-s*v[0]+c*v[1],v[2]])
        return velocity, np.r_[d.xpos[self.base],yaw]

    def step(self, command=(0.,0.,0.), push=None):
        d = self.data
        if d.time < 1.-1e-9:
            # Simulated support fixture holds the robot still for sensor-only calibration.
            d.qpos[:] = self.initial_qpos
            d.qvel[:] = 0.
            d.eq_active[self.fixture] = True
            self.policy.torque(d)
            mujoco.mj_step(self.model,d)
            d.qpos[:] = self.initial_qpos
            d.qvel[:] = 0.
            mujoco.mj_forward(self.model,d)
        else:
            d.eq_active[self.fixture] = False
            if self.step_index % 10 == 0:
                self.policy.infer(self.model,d,np.asarray(command))
            self.policy.torque(d,self.kp_scale,self.kd_scale)
            d.xfrc_applied[self.base,:3] = [0,0,0] if push is None else push
            mujoco.mj_step(self.model,d)
            mujoco.mj_forward(self.model,d)
        self.step_index += 1

    def fallen(self):
        return bool(self.data.xmat[self.base].reshape(3,3)[2,2] < .3 or self.data.qpos[2] < .32
                    or not np.isfinite(self.data.qpos).all())
