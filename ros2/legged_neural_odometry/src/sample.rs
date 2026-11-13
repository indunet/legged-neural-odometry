//! One 500 Hz sensor sample. Joint order is left then right, each
//! hip_pitch, hip_roll, hip_yaw, knee.

#[derive(Clone, Debug)]
pub struct SensorSample {
    pub t: f64,
    pub gyro: [f64; 3],
    pub accel: [f64; 3],
    pub q: [f64; 8],
    pub dq: [f64; 8],
    pub imu_yaw: f64,
    pub imu_yaw_t: f64,
}

impl SensorSample {
    pub fn sensors_finite(&self) -> bool {
        self.t.is_finite()
            && self.gyro.iter().all(|v| v.is_finite())
            && self.accel.iter().all(|v| v.is_finite())
            && self.q.iter().all(|v| v.is_finite())
            && self.dq.iter().all(|v| v.is_finite())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleFault {
    NonFinite,
    Timestamp,
    Yaw,
}

impl std::fmt::Display for SampleFault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFinite => write!(f, "non-finite sensor sample"),
            Self::Timestamp => write!(f, "sensor timestamps must increase with gaps <= 100 ms"),
            Self::Yaw => write!(f, "IMU yaw is missing or older than 20 ms"),
        }
    }
}
