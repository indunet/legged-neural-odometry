"""Public sensor-only inference API; does not import a simulator or training code."""
from collections import deque
import json
from pathlib import Path
import time
import numpy as np
import onnxruntime as ort

from .common import FEATURE_NAMES, ROOT
from .standstill import StandstillDetector
from .sensors import SensorSample, FeatureBuilder
from .uncertainty import PoseCovariance, QualityMonitor, epistemic_std, gate_keep, velocity_variance

MODEL = ROOT/'releases'/'estimator.onnx'


class Odometry:
    """Planar odometry: pose [x, y, yaw] relative to the start, velocity [vx, vy] in the heading frame.

    ``model`` is one ONNX path (default: the release) or a list of seed models
    combined at runtime. Heading is read from the IMU yaw. Covariances need a
    calibration (ONNX metadata or ``calibration``); otherwise ``velocity_cov``,
    ``pose_cov``, ``uncertainty`` and ``quality`` are None.
    """
    def __init__(self, model=None, calibration=None, yaw_drift_deg_min=None):
        paths=list(model) if isinstance(model,(list,tuple)) else [Path(model or MODEL)]
        options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
        self.sessions=[ort.InferenceSession(str(p),options,providers=['CPUExecutionProvider']) for p in paths]
        self.session=self.sessions[0]
        self.metadata=self.session.get_modelmeta().custom_metadata_map
        shape=self.session.get_inputs()[0].shape
        for session in self.sessions:
            meta=session.get_modelmeta().custom_metadata_map
            if tuple(meta['feature_names'].split(','))!=FEATURE_NAMES:
                raise ValueError('model feature order does not match the G1 sensor contract')
            if session.get_inputs()[0].shape[1:]!=shape[1:] or meta['label_names']!=self.metadata['label_names']:
                raise ValueError('ensemble members must share window and outputs')
        if tuple(self.metadata['label_names'].split(','))!=('vx','vy'):
            raise ValueError('expected outputs vx,vy')
        if not isinstance(shape[1],int) or shape[1]<1 or shape[2]!=len(FEATURE_NAMES):
            raise ValueError('model must take a fixed window of 25-feature frames')
        self.window=shape[1]
        names=tuple(o.name for o in self.session.get_outputs())
        self.variance_outputs={('velocity',):'none',('velocity','log_var'):'gaussian',
                               ('velocity','aleatoric_var','epistemic_var'):'ensemble'}.get(names)
        if self.variance_outputs is None or (len(self.sessions)>1 and self.variance_outputs=='ensemble'):
            raise ValueError(f'unsupported model outputs {names}')
        self.lag_gain=float(self.metadata['lag_compensation_s']) if self.metadata.get('label_frame')=='world_lowpass' else 0.
        if calibration is None and 'calibration' in self.metadata:calibration=json.loads(self.metadata['calibration'])
        elif isinstance(calibration,(str,Path)):calibration=json.loads(Path(calibration).read_text())
        self.calibration=calibration
        self.yaw_drift_deg_min=(yaw_drift_deg_min if yaw_drift_deg_min is not None else
                                (self.calibration or {}).get('yaw_drift_deg_min_default',1.))
        self.reset()

    def reset(self):
        self.builder=FeatureBuilder();self.frames=deque(maxlen=self.window);self.still=StandstillDetector()
        self.pose=np.zeros(3);self.previous_velocity=None;self.last_t=None
        self.last_yaw=None;self.relative_yaw=0.;self.latest=None;self.first_world_velocity=None
        if self.calibration:
            self.covariance=PoseCovariance(self.calibration,self.yaw_drift_deg_min)
            self.quality=QualityMonitor(self.calibration)

    def _infer(self, window):
        """Mean, per-axis aleatoric variance and seed-disagreement variance (or None)."""
        feed={'features':window}
        if len(self.sessions)==1:
            out=self.session.run(None,feed)
            if self.variance_outputs=='none':return out[0][0],None,None
            if self.variance_outputs=='gaussian':return out[0][0],np.exp(out[1][0]),np.zeros(2)
            return out[0][0],out[1][0],out[2][0]
        outs=[s.run(None,feed) for s in self.sessions]
        means=np.stack([o[0][0] for o in outs])
        aleatoric=(np.mean([np.exp(o[1][0]) for o in outs],axis=0) if self.variance_outputs=='gaussian'
                   else np.zeros(2))
        return means.mean(0),aleatoric,means.var(0)

    def update(self, sample:SensorSample):
        start=time.perf_counter()
        if sample.imu_yaw is None or not np.isfinite(sample.imu_yaw):
            raise ValueError('a finite IMU-provided yaw is required; no gyro-heading fallback')
        yaw_t=sample.t if sample.imu_yaw_t is None else sample.imu_yaw_t
        if not np.isfinite(yaw_t) or not -1e-9<=sample.t-yaw_t<=.02+1e-9:
            raise ValueError('IMU yaw timestamp must be current or at most 20 ms old')
        frame=self.builder.push(sample)
        if frame is None:return None
        self.frames.append(frame);quiet=self.still.update(frame)
        if len(self.frames)<self.window or not self.builder.attitude.ready:return None
        yaw=float(sample.imu_yaw)
        if self.last_yaw is None:self.last_yaw=yaw
        self.relative_yaw+=float(np.arctan2(np.sin(yaw-self.last_yaw),np.cos(yaw-self.last_yaw)))
        self.last_yaw=yaw
        predicted,aleatoric,epistemic=self._infer(np.stack(self.frames)[None])
        velocity=np.asarray(predicted,dtype=float)
        keep=float(gate_keep(velocity,self.still.value))
        velocity=self.still.gate_velocity(velocity)
        dt=0. if self.last_t is None else sample.t-self.last_t
        self.last_t=sample.t
        average=velocity if self.previous_velocity is None else .5*(velocity+self.previous_velocity)
        mid_yaw=.5*(self.pose[2]+self.relative_yaw)
        c,s=np.cos(mid_yaw),np.sin(mid_yaw)
        self.pose[:2]+=dt*np.array([c*average[0]-s*average[1],s*average[0]+c*average[1]])
        self.pose[2]=self.relative_yaw;self.previous_velocity=velocity.copy()
        pose=self.pose.copy()
        if self.lag_gain:
            c,s=np.cos(self.pose[2]),np.sin(self.pose[2])
            world=np.array([c*velocity[0]-s*velocity[1],s*velocity[0]+c*velocity[1]])
            if self.first_world_velocity is None:self.first_world_velocity=world
            pose[:2]+=self.lag_gain*(world-self.first_world_velocity)
        velocity_cov=pose_cov=uncertainty=quality=None
        if self.calibration and aleatoric is not None:
            cal=self.calibration
            variance=velocity_variance(aleatoric,epistemic,keep,cal)
            if dt>0:self.covariance.step(dt,mid_yaw,average,np.sqrt(variance))
            speed=float(epistemic_std(epistemic,cal))
            quality=self.quality.update(speed)
            velocity_cov=np.diag(variance).ravel().tolist()
            pose_cov=self.covariance.pose_cov.ravel().tolist()
            uncertainty={'aleatoric_std':(cal['aleatoric_scale']*np.sqrt(aleatoric)).tolist(),
                         'epistemic_std':(cal['epistemic_scale']*np.sqrt(epistemic)).tolist(),
                         'epistemic_speed_std':speed}
        self.latest={'t':float(sample.t),'velocity':velocity.tolist(),'pose':pose.tolist(),
                     'stationary':bool(quiet>.95),'stationary_degree':float(quiet),'ready':True,
                     'velocity_cov':velocity_cov,'pose_cov':pose_cov,'uncertainty':uncertainty,'quality':quality,
                     'latency_us':(time.perf_counter()-start)*1e6}
        return self.latest
