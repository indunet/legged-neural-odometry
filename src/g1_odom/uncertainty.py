"""Calibrated velocity covariance and pose covariance propagation; numpy only.

Shared by the runtime and the offline evaluation so both use identical numbers.
"""
import numpy as np

CHI2_2DOF_95 = 5.991464547107979
CHI2_2DOF_68 = 2.2788
QUALITY_LEVELS = ('ok', 'warn', 'alarm')


def gate_keep(velocity, degree, speed_full=.01, speed_off=.02):
    """The factor ``StandstillDetector.gate_velocity`` multiplies the velocity by."""
    speed = np.linalg.norm(np.asarray(velocity, dtype=float), axis=-1)
    weight = np.clip((speed_off-speed)/(speed_off-speed_full), 0., 1.)
    return 1.-np.asarray(degree, dtype=float)*weight


def velocity_variance(aleatoric, epistemic, keep, cal):
    """Per-axis calibrated variance; gated frames blend towards the standing variance."""
    keep = np.asarray(keep, dtype=float)[..., None]
    moving = cal['aleatoric_scale']**2*np.asarray(aleatoric)+cal['epistemic_scale']**2*np.asarray(epistemic)
    return keep**2*moving+(1.-keep**2)*np.asarray(cal['static_var'])+cal['var_floor']


def epistemic_std(epistemic, cal):
    """Calibrated seed-disagreement speed std, the quantity the quality flag watches."""
    return cal['epistemic_scale']*np.sqrt(np.sum(np.asarray(epistemic), axis=-1))


class QualityMonitor:
    """Exponentially smoothed epistemic std against validation-derived thresholds."""

    def __init__(self, cal, dt=.02):
        self.warn, self.alarm = cal['quality_warn'], cal['quality_alarm']
        self.alpha = dt/(cal.get('quality_tau_s', 1.)+dt)
        self.reset()

    def reset(self):
        self.value = None

    def update(self, epistemic_speed_std):
        x = float(epistemic_speed_std)
        self.value = x if self.value is None else self.value+self.alpha*(x-self.value)
        return 'alarm' if self.value > self.alarm else 'warn' if self.value > self.warn else 'ok'


class PoseCovariance:
    """Covariance of the error state [px, py, yaw, bx, by].

    b is the slowly varying velocity bias in units of the calibrated velocity std,
    with variance ``bias_fraction`` (a random constant when ``bias_time_s`` is
    None). The fast remainder decorrelates within ``correlation_time_s`` and
    enters position as white noise of spectral density 2 T_c (1 - rho) sigma^2.
    Yaw is a random walk whose 1-sigma reaches ``yaw_drift_deg_min`` after 60 s.
    """

    def __init__(self, cal, yaw_drift_deg_min=1.):
        self.rho = float(cal['bias_fraction'])
        self.tc = float(cal['correlation_time_s'])
        self.tb = cal.get('bias_time_s')
        self.scale = float(cal.get('pose_scale', 1.))
        self.q_yaw = np.deg2rad(yaw_drift_deg_min)**2/60.
        self.reset()

    def reset(self):
        self.P = np.zeros((5, 5))
        self.P[3:, 3:] = np.eye(2)*self.rho*self.scale

    def step(self, dt, yaw, velocity, std):
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s], [s, c]])
        vx, vy = float(velocity[0]), float(velocity[1])
        std = np.asarray(std, dtype=float)
        phi = 1. if self.tb is None else float(np.exp(-dt/self.tb))
        F = np.eye(5)
        F[:2, 2] = dt*np.array([-s*vx-c*vy, c*vx-s*vy])
        F[:2, 3:] = dt*R*std[None, :]
        F[3:, 3:] *= phi
        Q = np.zeros((5, 5))
        white = 2.*self.tc*(1.-self.rho)*self.scale*std**2
        Q[:2, :2] = (R*white[None, :])@R.T*dt
        Q[2, 2] = self.q_yaw*dt
        Q[3:, 3:] = np.eye(2)*self.rho*self.scale*(1.-phi**2)
        P = F@self.P@F.T+Q
        self.P = .5*(P+P.T)
        return self.P

    @property
    def pose_cov(self):
        return self.P[:3, :3].copy()
