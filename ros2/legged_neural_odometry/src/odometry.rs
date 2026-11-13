//! Planar odometry. Port of `g1_odom.runtime.Odometry`.

use std::collections::VecDeque;
use std::path::Path;

use ndarray::Array3;

use crate::features::FeatureBuilder;
use crate::sample::{SampleFault, SensorSample};
use crate::standstill::{keep_factor, StandstillDetector};
use crate::uncertainty::{
    epistemic_speed_std, velocity_variance, Calibration, PoseCovariance, Quality, QualityMonitor,
};

pub const FEATURE_NAMES: [&str; 25] = [
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "accel_x",
    "accel_y",
    "accel_z",
    "gravity_x",
    "gravity_y",
    "gravity_z",
    "q_left_hip_pitch_joint",
    "q_left_hip_roll_joint",
    "q_left_hip_yaw_joint",
    "q_left_knee_joint",
    "q_right_hip_pitch_joint",
    "q_right_hip_roll_joint",
    "q_right_hip_yaw_joint",
    "q_right_knee_joint",
    "dq_left_hip_pitch_joint",
    "dq_left_hip_roll_joint",
    "dq_left_hip_yaw_joint",
    "dq_left_knee_joint",
    "dq_right_hip_pitch_joint",
    "dq_right_hip_roll_joint",
    "dq_right_hip_yaw_joint",
    "dq_right_knee_joint",
];

const WINDOW: usize = 25;

#[derive(Clone, Debug)]
pub struct OdometryResult {
    pub t: f64,
    pub velocity: [f64; 2],
    pub pose: [f64; 3],
    pub gyro_z: f64,
    pub stationary: bool,
    pub stationary_degree: f64,
    pub velocity_cov: [f64; 4],
    pub pose_cov: [f64; 9],
    pub epistemic_speed_std: f64,
    pub quality: Quality,
}

#[derive(Debug)]
pub enum OdometryError {
    Model(String),
    Sample(SampleFault),
}

impl std::fmt::Display for OdometryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Model(message) => write!(f, "{message}"),
            Self::Sample(fault) => write!(f, "{fault}"),
        }
    }
}

impl std::error::Error for OdometryError {}

impl From<SampleFault> for OdometryError {
    fn from(fault: SampleFault) -> Self {
        Self::Sample(fault)
    }
}

pub struct Odometry {
    session: ort::session::Session,
    calibration: Calibration,
    lag_gain: f64,
    yaw_drift_deg_min: f64,
    builder: FeatureBuilder,
    frames: VecDeque<[f32; 25]>,
    still: StandstillDetector,
    pose: [f64; 3],
    previous_velocity: Option<[f64; 2]>,
    last_t: Option<f64>,
    last_yaw: Option<f64>,
    relative_yaw: f64,
    first_world_velocity: Option<[f64; 2]>,
    covariance: PoseCovariance,
    quality: QualityMonitor,
}

impl Odometry {
    pub fn open(
        model: &Path,
        onnxruntime: &Path,
        yaw_drift_deg_min: Option<f64>,
    ) -> Result<Self, OdometryError> {
        let environment = runtime(onnxruntime)?;
        let session = ort::session::Session::builder(environment)
            .map_err(model_err)?
            .with_intra_threads(1)
            .map_err(model_err)?
            .with_inter_threads(1)
            .map_err(model_err)?
            .commit_from_file(model)
            .map_err(model_err)?;
        let (lag_gain, calibration) = {
            let metadata = session.metadata().map_err(model_err)?;
            let feature_names = metadata_string(&metadata, "feature_names")?;
            if feature_names.split(',').collect::<Vec<_>>() != FEATURE_NAMES {
                return Err(OdometryError::Model(
                    "model feature order does not match the G1 sensor contract".into(),
                ));
            }
            let labels = metadata_string(&metadata, "label_names")?;
            if labels != "vx,vy" {
                return Err(OdometryError::Model(format!(
                    "expected outputs vx,vy, got {labels}"
                )));
            }
            let window: usize = metadata_string(&metadata, "window")?
                .parse()
                .map_err(model_err)?;
            if window != WINDOW {
                return Err(OdometryError::Model(format!(
                    "expected a window of {WINDOW}, got {window}"
                )));
            }
            let lag_gain = if metadata_string(&metadata, "label_frame")? == "world_lowpass" {
                metadata_string(&metadata, "lag_compensation_s")?
                    .parse()
                    .map_err(model_err)?
            } else {
                0.0
            };
            let calibration: Calibration =
                serde_json::from_str(&metadata_string(&metadata, "calibration")?)
                    .map_err(model_err)?;
            (lag_gain, calibration)
        };
        let yaw_drift = yaw_drift_deg_min.unwrap_or(calibration.yaw_drift_deg_min_default);
        let mut odom = Self {
            session,
            lag_gain,
            yaw_drift_deg_min: yaw_drift,
            covariance: PoseCovariance::new(&calibration, yaw_drift),
            quality: QualityMonitor::new(&calibration),
            calibration,
            builder: FeatureBuilder::new(),
            frames: VecDeque::with_capacity(WINDOW),
            still: StandstillDetector::new(),
            pose: [0.0; 3],
            previous_velocity: None,
            last_t: None,
            last_yaw: None,
            relative_yaw: 0.0,
            first_world_velocity: None,
        };
        odom.reset_state();
        Ok(odom)
    }

    pub fn reset(&mut self) {
        self.reset_state();
    }

    fn reset_state(&mut self) {
        self.builder = FeatureBuilder::new();
        self.frames.clear();
        self.still = StandstillDetector::new();
        self.pose = [0.0; 3];
        self.previous_velocity = None;
        self.last_t = None;
        self.last_yaw = None;
        self.relative_yaw = 0.0;
        self.first_world_velocity = None;
        self.covariance = PoseCovariance::new(&self.calibration, self.yaw_drift_deg_min);
        self.quality = QualityMonitor::new(&self.calibration);
    }

    pub fn update(
        &mut self,
        sample: &SensorSample,
    ) -> Result<Option<OdometryResult>, OdometryError> {
        if !sample.imu_yaw.is_finite() {
            return Err(SampleFault::Yaw.into());
        }
        let yaw_age = sample.t - sample.imu_yaw_t;
        if !sample.imu_yaw_t.is_finite() || !(-1e-9..=0.02 + 1e-9).contains(&yaw_age) {
            return Err(SampleFault::Yaw.into());
        }
        let frame = self.builder.push(sample)?;
        let Some(frame) = frame else { return Ok(None) };
        if self.frames.len() == WINDOW {
            self.frames.pop_front();
        }
        self.frames.push_back(frame);
        let quiet = self.still.update(&frame);
        if self.frames.len() < WINDOW || !self.builder.attitude.ready {
            return Ok(None);
        }

        let yaw = sample.imu_yaw;
        if self.last_yaw.is_none() {
            self.last_yaw = Some(yaw);
        }
        let delta = yaw - self.last_yaw.unwrap();
        self.relative_yaw += delta.sin().atan2(delta.cos());
        self.last_yaw = Some(yaw);

        let (predicted, aleatoric, epistemic) = self.infer()?;
        let keep = keep_factor(predicted, self.still.value);
        let velocity = [predicted[0] * keep, predicted[1] * keep];
        let dt = match self.last_t {
            None => 0.0,
            Some(previous) => sample.t - previous,
        };
        self.last_t = Some(sample.t);
        let average = match self.previous_velocity {
            None => velocity,
            Some(previous) => [
                (velocity[0] + previous[0]) * 0.5,
                (velocity[1] + previous[1]) * 0.5,
            ],
        };
        let mid_yaw = 0.5 * (self.pose[2] + self.relative_yaw);
        let (c, s) = (mid_yaw.cos(), mid_yaw.sin());
        self.pose[0] += dt * (c * average[0] - s * average[1]);
        self.pose[1] += dt * (s * average[0] + c * average[1]);
        self.pose[2] = self.relative_yaw;
        self.previous_velocity = Some(velocity);

        let mut pose = self.pose;
        if self.lag_gain != 0.0 {
            let (c, s) = (self.pose[2].cos(), self.pose[2].sin());
            let world = [
                c * velocity[0] - s * velocity[1],
                s * velocity[0] + c * velocity[1],
            ];
            if self.first_world_velocity.is_none() {
                self.first_world_velocity = Some(world);
            }
            let first = self.first_world_velocity.unwrap();
            pose[0] += self.lag_gain * (world[0] - first[0]);
            pose[1] += self.lag_gain * (world[1] - first[1]);
        }

        let variance = velocity_variance(aleatoric, epistemic, keep, &self.calibration);
        if dt > 0.0 {
            self.covariance.step(
                dt,
                mid_yaw,
                average,
                [variance[0].sqrt(), variance[1].sqrt()],
            );
        }
        let speed = epistemic_speed_std(epistemic, &self.calibration);
        let quality = self.quality.update(speed);
        Ok(Some(OdometryResult {
            t: sample.t,
            velocity,
            pose,
            gyro_z: sample.gyro[2],
            stationary: quiet > 0.95,
            stationary_degree: quiet,
            velocity_cov: [variance[0], 0.0, 0.0, variance[1]],
            pose_cov: self.covariance.pose_cov(),
            epistemic_speed_std: speed,
            quality,
        }))
    }

    fn infer(&mut self) -> Result<([f64; 2], [f64; 2], [f64; 2]), OdometryError> {
        let mut data = Vec::with_capacity(WINDOW * 25);
        for frame in &self.frames {
            data.extend_from_slice(frame);
        }
        let array = Array3::from_shape_vec((1, WINDOW, 25), data).map_err(model_err)?;
        let tensor = ort::value::Tensor::from_array(array).map_err(model_err)?;
        let outputs = self
            .session
            .run(ort::inputs!["features" => tensor])
            .map_err(model_err)?;
        let velocity = tensor2(&outputs, "velocity")?;
        let aleatoric = tensor2(&outputs, "aleatoric_var")?;
        let epistemic = tensor2(&outputs, "epistemic_var")?;
        Ok((
            [velocity[0] as f64, velocity[1] as f64],
            [aleatoric[0] as f64, aleatoric[1] as f64],
            [epistemic[0] as f64, epistemic[1] as f64],
        ))
    }
}

fn model_err(error: impl std::fmt::Display) -> OdometryError {
    OdometryError::Model(error.to_string())
}

fn metadata_string(
    metadata: &ort::session::ModelMetadata,
    key: &str,
) -> Result<String, OdometryError> {
    metadata
        .custom(key)
        .ok_or_else(|| OdometryError::Model(format!("missing ONNX metadata {key}")))
}

fn tensor2(outputs: &ort::session::SessionOutputs, name: &str) -> Result<[f32; 2], OdometryError> {
    let (_shape, values) = outputs[name]
        .try_extract_tensor::<f32>()
        .map_err(model_err)?;
    if values.len() < 2 {
        return Err(OdometryError::Model(format!(
            "{name} did not return two values"
        )));
    }
    Ok([values[0], values[1]])
}

/// Locate `libonnxruntime.so`: `ORT_DYLIB_PATH`, then beside an installed
/// binary, then the usual system prefixes.
pub fn onnxruntime_library() -> Option<std::path::PathBuf> {
    if let Some(path) = std::env::var_os("ORT_DYLIB_PATH") {
        let path = std::path::PathBuf::from(path);
        if path.is_file() {
            return Some(path);
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(lib_dir) = exe.parent().and_then(|dir| dir.parent()) {
            let bundled = lib_dir.join("libonnxruntime.so");
            if bundled.is_file() {
                return Some(bundled);
            }
        }
    }
    [
        "/opt/legged-neural-odometry/lib/libonnxruntime.so",
        "/opt/onnxruntime/onnxruntime-linux-aarch64-1.31.0/lib/libonnxruntime.so",
        "/opt/onnxruntime/onnxruntime-linux-x64-1.31.0/lib/libonnxruntime.so",
    ]
    .into_iter()
    .map(std::path::PathBuf::from)
    .find(|path| path.is_file())
}

fn runtime(path: &Path) -> Result<&'static ort::environment::Environment, OdometryError> {
    use std::sync::{Mutex, OnceLock};
    static ENV: OnceLock<ort::environment::Environment> = OnceLock::new();
    static LOCK: Mutex<()> = Mutex::new(());
    if let Some(environment) = ENV.get() {
        return Ok(environment);
    }
    let _guard = LOCK.lock().unwrap_or_else(|poison| poison.into_inner());
    if let Some(environment) = ENV.get() {
        return Ok(environment);
    }
    let environment = ort::init_from(path)
        .map_err(model_err)?
        .build()
        .map_err(model_err)?;
    let _ = ENV.set(environment);
    Ok(ENV.get().expect("ONNX Runtime environment was just stored"))
}
