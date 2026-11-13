use std::path::PathBuf;

use legged_neural_odometry::{onnxruntime_library, Odometry, SensorSample};
use serde::Deserialize;

#[derive(Deserialize)]
struct Golden {
    samples: Vec<GoldenSample>,
    results: Vec<Option<GoldenResult>>,
}

#[derive(Deserialize)]
struct GoldenSample {
    t: f64,
    gyro: [f64; 3],
    accel: [f64; 3],
    q: [f64; 8],
    dq: [f64; 8],
    imu_yaw: f64,
    imu_yaw_t: f64,
}

#[derive(Deserialize)]
struct GoldenResult {
    t: f64,
    velocity: [f64; 2],
    pose: [f64; 3],
    stationary: bool,
    stationary_degree: f64,
    velocity_cov: [f64; 4],
    pose_cov: [f64; 9],
    quality: String,
    epistemic_speed_std: f64,
}

fn onnxruntime() -> PathBuf {
    onnxruntime_library().expect("libonnxruntime.so")
}

fn model() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../releases/estimator.onnx")
}

fn sample_at(index: usize, yaw: f64) -> SensorSample {
    SensorSample {
        t: index as f64 * 0.002,
        gyro: [0.0; 3],
        accel: [0.0, 0.0, 9.81],
        q: [0.0; 8],
        dq: [0.0; 8],
        imu_yaw: yaw,
        imu_yaw_t: index as f64 * 0.002,
    }
}

#[test]
fn matches_python_odometry() {
    let golden: Golden = serde_json::from_str(include_str!("golden.json")).expect("golden json");
    assert_eq!(golden.samples.len(), golden.results.len());
    let mut odom = Odometry::open(&model(), &onnxruntime(), None).expect("open model");
    let mut ready = 0;
    let mut worst = [0.0_f64; 6];
    for (sample, expected) in golden.samples.iter().zip(&golden.results) {
        let got = odom
            .update(&SensorSample {
                t: sample.t,
                gyro: sample.gyro,
                accel: sample.accel,
                q: sample.q,
                dq: sample.dq,
                imu_yaw: sample.imu_yaw,
                imu_yaw_t: sample.imu_yaw_t,
            })
            .expect("sample");
        match (got, expected) {
            (None, None) => {}
            (Some(got), Some(expected)) => {
                ready += 1;
                worst[0] = worst[0].max((got.t - expected.t).abs());
                worst[1] = worst[1].max(max_abs2(got.velocity, expected.velocity));
                worst[2] = worst[2].max(max_abs3(got.pose, expected.pose));
                worst[3] = worst[3].max((got.stationary_degree - expected.stationary_degree).abs());
                worst[4] = worst[4].max(max_abs4(got.velocity_cov, expected.velocity_cov));
                worst[5] = worst[5].max(
                    max_abs9(got.pose_cov, expected.pose_cov)
                        .max((got.epistemic_speed_std - expected.epistemic_speed_std).abs()),
                );
                assert_eq!(got.stationary, expected.stationary, "t={}", got.t);
                assert_eq!(got.quality.as_str(), expected.quality, "t={}", got.t);
            }
            (got, expected) => panic!(
                "ready mismatch at t={}: rust {got:?} python {}",
                sample.t,
                expected.is_some()
            ),
        }
    }
    assert!(
        ready > 50,
        "expected the window to become ready, got {ready}"
    );
    assert!(worst[1] <= 1e-5, "velocity {}", worst[1]);
    assert!(worst[2] <= 1e-5, "pose {}", worst[2]);
    assert!(worst[3] <= 1e-6, "standstill {}", worst[3]);
    assert!(worst[4] <= 1e-6, "velocity cov {}", worst[4]);
    assert!(worst[5] <= 1e-6, "pose cov {}", worst[5]);
}

#[test]
fn rejects_a_gap_without_advancing_time() {
    let mut odom = Odometry::open(&model(), &onnxruntime(), None).expect("open model");
    odom.update(&sample_at(0, 0.0)).unwrap();
    let err = odom.update(&sample_at(100, 0.0)).unwrap_err();
    assert!(err.to_string().contains("100 ms"), "{err}");
    odom.update(&sample_at(1, 0.0))
        .expect("timestamp was not consumed by the rejected sample");
}

#[test]
fn rejects_a_stale_yaw_without_advancing_time() {
    let mut odom = Odometry::open(&model(), &onnxruntime(), None).expect("open model");
    let mut stale = sample_at(0, 0.1);
    stale.imu_yaw_t = -0.05;
    assert!(odom.update(&stale).unwrap_err().to_string().contains("yaw"));
    odom.update(&sample_at(0, 0.1))
        .expect("rejected yaw must not consume the timestamp");
}

fn max_abs2(got: [f64; 2], expected: [f64; 2]) -> f64 {
    got.into_iter()
        .zip(expected)
        .map(|(got, expected)| (got - expected).abs())
        .fold(0.0, f64::max)
}

fn max_abs3(got: [f64; 3], expected: [f64; 3]) -> f64 {
    got.into_iter()
        .zip(expected)
        .map(|(got, expected)| (got - expected).abs())
        .fold(0.0, f64::max)
}

fn max_abs4(got: [f64; 4], expected: [f64; 4]) -> f64 {
    got.into_iter()
        .zip(expected)
        .map(|(got, expected)| (got - expected).abs())
        .fold(0.0, f64::max)
}

fn max_abs9(got: [f64; 9], expected: [f64; 9]) -> f64 {
    got.into_iter()
        .zip(expected)
        .map(|(got, expected)| (got - expected).abs())
        .fold(0.0, f64::max)
}
