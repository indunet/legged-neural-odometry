//! Standstill degree from feature frames. Port of `StandstillDetector`.

const WINDOW: usize = 25;
const FRAME_RATE: f64 = 50.0;
const TAU: f64 = 0.2;
const DQ_MAX: f64 = 0.12;
const GYRO_MAX: f64 = 0.06;
const ACCEL_STD_MAX: f64 = 0.10;

pub struct StandstillDetector {
    history: Vec<(f64, f64, f64)>,
    pub value: f64,
}

impl StandstillDetector {
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
            value: 0.0,
        }
    }

    pub fn update(&mut self, frame: &[f32; 25]) -> f64 {
        let mut max_dq = 0.0_f64;
        for value in &frame[17..25] {
            max_dq = max_dq.max((*value as f64).abs());
        }
        let gyro = (frame[0] as f64)
            .hypot(frame[1] as f64)
            .hypot(frame[2] as f64);
        let accel = (frame[3] as f64)
            .hypot(frame[4] as f64)
            .hypot(frame[5] as f64);
        self.history.push((max_dq, gyro, accel));
        if self.history.len() > WINDOW {
            let extra = self.history.len() - WINDOW;
            self.history.drain(0..extra);
        }
        let quiet = self.history.len() == WINDOW
            && self.history.iter().all(|row| row.0 < DQ_MAX)
            && self.history.iter().all(|row| row.1 < GYRO_MAX)
            && population_std(self.history.iter().map(|row| row.2)) < ACCEL_STD_MAX;
        if quiet {
            let step = 1.0 - (-1.0 / FRAME_RATE / TAU).exp();
            self.value += step * (1.0 - self.value);
        } else {
            self.value = 0.0;
        }
        self.value
    }
}

fn population_std(values: impl Iterator<Item = f64> + Clone) -> f64 {
    let n = values.clone().count() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let mean = values.clone().sum::<f64>() / n;
    (values.map(|value| (value - mean).powi(2)).sum::<f64>() / n).sqrt()
}

pub fn keep_factor(velocity: [f64; 2], degree: f64) -> f64 {
    let speed = velocity[0].hypot(velocity[1]);
    let weight = ((0.02 - speed) / (0.02 - 0.01)).clamp(0.0, 1.0);
    1.0 - degree * weight
}
