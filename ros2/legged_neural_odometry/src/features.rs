//! 500 Hz samples to 50 Hz feature frames. Port of `FeatureBuilder`.

use crate::attitude::Attitude;
use crate::sample::{SampleFault, SensorSample};

pub const FRAME_DT: f64 = 0.02;

pub struct FeatureBuilder {
    pub attitude: Attitude,
    rows: Vec<[f64; 6]>,
    last_frame_t: Option<f64>,
}

impl FeatureBuilder {
    pub fn new() -> Self {
        Self {
            attitude: Attitude::new(),
            rows: Vec::new(),
            last_frame_t: None,
        }
    }

    pub fn push(&mut self, sample: &SensorSample) -> Result<Option<[f32; 25]>, SampleFault> {
        let gravity = self.attitude.update(sample)?;
        self.rows.push([
            sample.gyro[0],
            sample.gyro[1],
            sample.gyro[2],
            sample.accel[0],
            sample.accel[1],
            sample.accel[2],
        ]);
        if self.last_frame_t.is_none() {
            self.last_frame_t = Some(sample.t);
        }
        if sample.t - self.last_frame_t.unwrap() < FRAME_DT - 1e-8 {
            return Ok(None);
        }
        self.last_frame_t = Some(sample.t);
        let n = self.rows.len() as f64;
        let mut mean = [0.0; 6];
        for row in &self.rows {
            for (slot, value) in mean.iter_mut().zip(row) {
                *slot += *value;
            }
        }
        self.rows.clear();
        let mut frame = [0.0_f32; 25];
        for (i, value) in mean.iter().enumerate() {
            frame[i] = (*value / n) as f32;
        }
        for i in 0..3 {
            frame[6 + i] = gravity[i] as f32;
        }
        for i in 0..8 {
            frame[9 + i] = sample.q[i] as f32;
            frame[17 + i] = sample.dq[i] as f32;
        }
        Ok(Some(frame))
    }
}
