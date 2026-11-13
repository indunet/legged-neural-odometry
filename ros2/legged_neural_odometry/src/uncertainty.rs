//! Calibrated velocity variance and the 5-state pose covariance.
//! Port of `g1_odom.uncertainty`.

use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
pub struct Calibration {
    pub aleatoric_scale: f64,
    pub epistemic_scale: f64,
    pub static_var: [f64; 2],
    pub var_floor: f64,
    pub correlation_time_s: f64,
    pub bias_fraction: f64,
    #[serde(default)]
    pub bias_time_s: Option<f64>,
    #[serde(default = "default_pose_scale")]
    pub pose_scale: f64,
    #[serde(default = "default_tau")]
    pub quality_tau_s: f64,
    pub quality_warn: f64,
    pub quality_alarm: f64,
    #[serde(default = "default_yaw_drift")]
    pub yaw_drift_deg_min_default: f64,
}

fn default_pose_scale() -> f64 {
    1.0
}
fn default_tau() -> f64 {
    1.0
}
fn default_yaw_drift() -> f64 {
    1.0
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quality {
    Ok,
    Warn,
    Alarm,
}

impl Quality {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Warn => "warn",
            Self::Alarm => "alarm",
        }
    }
}

pub struct QualityMonitor {
    warn: f64,
    alarm: f64,
    alpha: f64,
    value: Option<f64>,
}

impl QualityMonitor {
    pub fn new(cal: &Calibration) -> Self {
        const DT: f64 = 0.02;
        Self {
            warn: cal.quality_warn,
            alarm: cal.quality_alarm,
            alpha: DT / (cal.quality_tau_s + DT),
            value: None,
        }
    }

    pub fn update(&mut self, epistemic_speed_std: f64) -> Quality {
        self.value = Some(match self.value {
            None => epistemic_speed_std,
            Some(previous) => previous + self.alpha * (epistemic_speed_std - previous),
        });
        let value = self.value.unwrap();
        if value > self.alarm {
            Quality::Alarm
        } else if value > self.warn {
            Quality::Warn
        } else {
            Quality::Ok
        }
    }
}

pub fn velocity_variance(
    aleatoric: [f64; 2],
    epistemic: [f64; 2],
    keep: f64,
    cal: &Calibration,
) -> [f64; 2] {
    let keep2 = keep * keep;
    let mut out = [0.0; 2];
    for i in 0..2 {
        let moving =
            cal.aleatoric_scale.powi(2) * aleatoric[i] + cal.epistemic_scale.powi(2) * epistemic[i];
        out[i] = keep2 * moving + (1.0 - keep2) * cal.static_var[i] + cal.var_floor;
    }
    out
}

pub fn epistemic_speed_std(epistemic: [f64; 2], cal: &Calibration) -> f64 {
    cal.epistemic_scale * (epistemic[0] + epistemic[1]).sqrt()
}

pub struct PoseCovariance {
    rho: f64,
    tc: f64,
    tb: Option<f64>,
    scale: f64,
    q_yaw: f64,
    p: [[f64; 5]; 5],
}

impl PoseCovariance {
    pub fn new(cal: &Calibration, yaw_drift_deg_min: f64) -> Self {
        let mut filter = Self {
            rho: cal.bias_fraction,
            tc: cal.correlation_time_s,
            tb: cal.bias_time_s,
            scale: cal.pose_scale,
            q_yaw: yaw_drift_deg_min.to_radians().powi(2) / 60.0,
            p: [[0.0; 5]; 5],
        };
        filter.reset();
        filter
    }

    pub fn reset(&mut self) {
        self.p = [[0.0; 5]; 5];
        let bias = self.rho * self.scale;
        self.p[3][3] = bias;
        self.p[4][4] = bias;
    }

    pub fn step(&mut self, dt: f64, yaw: f64, velocity: [f64; 2], std: [f64; 2]) {
        let (c, s) = (yaw.cos(), yaw.sin());
        let (vx, vy) = (velocity[0], velocity[1]);
        let phi = match self.tb {
            None => 1.0,
            Some(tb) => (-dt / tb).exp(),
        };
        let mut f = identity5();
        f[0][2] = dt * (-s * vx - c * vy);
        f[1][2] = dt * (c * vx - s * vy);
        f[0][3] = dt * c * std[0];
        f[0][4] = dt * -s * std[1];
        f[1][3] = dt * s * std[0];
        f[1][4] = dt * c * std[1];
        f[3][3] = phi;
        f[4][4] = phi;

        let white = [
            2.0 * self.tc * (1.0 - self.rho) * self.scale * std[0] * std[0],
            2.0 * self.tc * (1.0 - self.rho) * self.scale * std[1] * std[1],
        ];
        let m = [[c * white[0], -s * white[1]], [s * white[0], c * white[1]]];
        let mut q = [[0.0; 5]; 5];
        q[0][0] = (m[0][0] * c + m[0][1] * -s) * dt;
        q[0][1] = (m[0][0] * s + m[0][1] * c) * dt;
        q[1][0] = (m[1][0] * c + m[1][1] * -s) * dt;
        q[1][1] = (m[1][0] * s + m[1][1] * c) * dt;
        q[2][2] = self.q_yaw * dt;
        let bias_noise = self.rho * self.scale * (1.0 - phi * phi);
        q[3][3] = bias_noise;
        q[4][4] = bias_noise;

        let fp = matmul5(&f, &self.p);
        let ft = transpose5(&f);
        let mut next = matmul5(&fp, &ft);
        for i in 0..5 {
            for j in 0..5 {
                next[i][j] += q[i][j];
            }
        }
        for i in 0..5 {
            for j in i..5 {
                let mean = 0.5 * (next[i][j] + next[j][i]);
                next[i][j] = mean;
                next[j][i] = mean;
            }
        }
        self.p = next;
    }

    pub fn pose_cov(&self) -> [f64; 9] {
        let mut out = [0.0; 9];
        for row in 0..3 {
            for col in 0..3 {
                out[row * 3 + col] = self.p[row][col];
            }
        }
        out
    }
}

fn identity5() -> [[f64; 5]; 5] {
    let mut m = [[0.0; 5]; 5];
    for i in 0..5 {
        m[i][i] = 1.0;
    }
    m
}

fn matmul5(a: &[[f64; 5]; 5], b: &[[f64; 5]; 5]) -> [[f64; 5]; 5] {
    let mut out = [[0.0; 5]; 5];
    for i in 0..5 {
        for k in 0..5 {
            let aik = a[i][k];
            for j in 0..5 {
                out[i][j] += aik * b[k][j];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::ros_covariance;

    #[test]
    fn covariance_uses_x_y_and_yaw_slots() {
        let pose = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let velocity = [10.0, 11.0, 12.0, 13.0];
        let (pose36, twist36) = ros_covariance(&pose, &velocity);
        assert_eq!(pose36[0], 1.0);
        assert_eq!(pose36[1], 2.0);
        assert_eq!(pose36[5], 3.0);
        assert_eq!(pose36[6], 4.0);
        assert_eq!(pose36[7], 5.0);
        assert_eq!(pose36[11], 6.0);
        assert_eq!(pose36[30], 7.0);
        assert_eq!(pose36[31], 8.0);
        assert_eq!(pose36[35], 9.0);
        assert_eq!(twist36[0], 10.0);
        assert_eq!(twist36[1], 11.0);
        assert_eq!(twist36[6], 12.0);
        assert_eq!(twist36[7], 13.0);
        assert_eq!(pose36.iter().filter(|value| **value != 0.0).count(), 9);
    }
}

fn transpose5(a: &[[f64; 5]; 5]) -> [[f64; 5]; 5] {
    let mut out = [[0.0; 5]; 5];
    for i in 0..5 {
        for j in 0..5 {
            out[j][i] = a[i][j];
        }
    }
    out
}

/// Map the planar covariances into `nav_msgs/Odometry` 6×6 blocks.
/// Pose order is x, y, yaw. Velocity order is vx, vy.
pub fn ros_covariance(pose_cov: &[f64; 9], velocity_cov: &[f64; 4]) -> ([f64; 36], [f64; 36]) {
    const POSE_SLOT: [usize; 9] = [0, 1, 5, 6, 7, 11, 30, 31, 35];
    let mut pose = [0.0; 36];
    for (value, slot) in pose_cov.iter().zip(POSE_SLOT) {
        pose[slot] = *value;
    }
    const VELOCITY_SLOT: [usize; 4] = [0, 1, 6, 7];
    let mut twist = [0.0; 36];
    for (value, slot) in velocity_cov.iter().zip(VELOCITY_SLOT) {
        twist[slot] = *value;
    }
    (pose, twist)
}
