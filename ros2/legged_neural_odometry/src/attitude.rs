//! Complementary gravity filter. Port of `g1_odom.sensors.Attitude`.

use crate::rotation::{align_up_to_world, from_rotvec, Mat3};
use crate::sample::{SampleFault, SensorSample};

const DOWN: [f64; 3] = [0.0, 0.0, -1.0];
const WORLD_UP: [f64; 3] = [0.0, 0.0, 1.0];

pub struct Attitude {
    rotation: Mat3,
    bias: Option<[f64; 3]>,
    t: Option<f64>,
    calibration: Vec<[f64; 6]>,
    calibration_start: Option<f64>,
    accel_lp: Option<[f64; 3]>,
    pub ready: bool,
}

impl Attitude {
    pub fn new() -> Self {
        Self {
            rotation: Mat3::identity(),
            bias: None,
            t: None,
            calibration: Vec::new(),
            calibration_start: None,
            accel_lp: None,
            ready: false,
        }
    }

    pub fn update(&mut self, sample: &SensorSample) -> Result<[f64; 3], SampleFault> {
        if !sample.sensors_finite() {
            return Err(SampleFault::NonFinite);
        }
        let dt = match self.t {
            None => 0.0,
            Some(previous) => sample.t - previous,
        };
        if dt < 0.0 || dt > 0.1 {
            return Err(SampleFault::Timestamp);
        }
        if self.t.is_some() && dt == 0.0 {
            return Err(SampleFault::Timestamp);
        }
        self.t = Some(sample.t);

        if self.bias.is_none() {
            let gyro_norm = sample.gyro[0].hypot(sample.gyro[1]).hypot(sample.gyro[2]);
            let accel_norm = sample.accel[0]
                .hypot(sample.accel[1])
                .hypot(sample.accel[2]);
            let max_dq = sample.dq.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            let quiet = max_dq < 0.08 && gyro_norm < 0.08 && (accel_norm - 9.81).abs() < 0.65;
            if !quiet {
                self.calibration.clear();
                self.calibration_start = None;
                return Ok(DOWN);
            }
            if self.calibration_start.is_none() {
                self.calibration_start = Some(sample.t);
            }
            self.calibration.push([
                sample.gyro[0],
                sample.gyro[1],
                sample.gyro[2],
                sample.accel[0],
                sample.accel[1],
                sample.accel[2],
            ]);
            if self.calibration.len() > 400 {
                let extra = self.calibration.len() - 400;
                self.calibration.drain(0..extra);
            }
            let elapsed = sample.t - self.calibration_start.unwrap();
            if elapsed >= 0.75 && self.calibration.len() >= 50 {
                let gyro_std = column_std(&self.calibration, 0);
                let accel_std = column_std(&self.calibration, 3);
                if gyro_std.into_iter().fold(0.0_f64, f64::max) < 0.02
                    && accel_std.into_iter().fold(0.0_f64, f64::max) < 0.25
                {
                    let mean = column_mean(&self.calibration);
                    self.bias = Some([mean[0], mean[1], mean[2]]);
                    let up = [mean[3], mean[4], mean[5]];
                    self.rotation = align_up_to_world(up);
                    self.ready = true;
                }
            }
            return Ok(self.rotation.transpose_mul_vec(DOWN));
        }

        let bias = self.bias.unwrap();
        let omega = [
            sample.gyro[0] - bias[0],
            sample.gyro[1] - bias[1],
            sample.gyro[2] - bias[2],
        ];
        self.rotation = self.rotation.mul(from_rotvec(scale(omega, dt)));
        let accel = sample.accel;
        self.accel_lp = Some(match self.accel_lp {
            None => accel,
            Some(previous) => {
                let alpha = dt / (0.35 + dt);
                [
                    previous[0] + alpha * (accel[0] - previous[0]),
                    previous[1] + alpha * (accel[1] - previous[1]),
                    previous[2] + alpha * (accel[2] - previous[2]),
                ]
            }
        });
        let accel_lp = self.accel_lp.unwrap();
        let accel_norm = accel_lp[0].hypot(accel_lp[1]).hypot(accel_lp[2]);
        let confidence = (1.0 - (accel_norm - 9.81).abs() / 1.5).clamp(0.0, 1.0);
        let measured_up = scale(accel_lp, 1.0 / accel_norm.max(1e-9));
        let predicted_up = self.rotation.transpose_mul_vec(WORLD_UP);
        let correction = scale(cross(measured_up, predicted_up), 0.35 * confidence);
        self.rotation = self.rotation.mul(from_rotvec(scale(correction, dt)));
        Ok(self.rotation.transpose_mul_vec(DOWN))
    }
}

fn scale(v: [f64; 3], s: f64) -> [f64; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn column_mean(rows: &[[f64; 6]]) -> [f64; 6] {
    let mut mean = [0.0; 6];
    for row in rows {
        for (slot, value) in mean.iter_mut().zip(row) {
            *slot += *value;
        }
    }
    let n = rows.len() as f64;
    for value in &mut mean {
        *value /= n;
    }
    mean
}

/// Population standard deviation, matching `numpy.std` (`ddof=0`).
fn column_std(rows: &[[f64; 6]], start: usize) -> [f64; 3] {
    let mean = column_mean(rows);
    let mut var = [0.0; 3];
    for row in rows {
        for k in 0..3 {
            let delta = row[start + k] - mean[start + k];
            var[k] += delta * delta;
        }
    }
    let n = rows.len() as f64;
    [
        (var[0] / n).sqrt(),
        (var[1] / n).sqrt(),
        (var[2] / n).sqrt(),
    ]
}
