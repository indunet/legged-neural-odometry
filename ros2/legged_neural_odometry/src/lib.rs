//! Neural planar odometry for the Unitree G1.
//!
//! The numeric path matches `g1_odom.runtime.Odometry`: 500 Hz IMU and eight
//! hip/knee joints become 50 Hz features, `releases/estimator.onnx` predicts
//! heading-frame velocity, and IMU yaw integrates that velocity.

mod attitude;
mod features;
mod odometry;
mod rotation;
mod sample;
mod standstill;
mod uncertainty;

pub use odometry::{onnxruntime_library, Odometry, OdometryError, OdometryResult, FEATURE_NAMES};
pub use sample::{SampleFault, SensorSample};
pub use uncertainty::{ros_covariance, Quality};
