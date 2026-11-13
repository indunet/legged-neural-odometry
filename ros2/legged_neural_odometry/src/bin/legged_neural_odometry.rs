//! ROS 2 node: `/lowstate` in, `/g1/odom` out.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::{executor::LocalPool, task::LocalSpawnExt, StreamExt};
use legged_neural_odometry::{
    onnxruntime_library, ros_covariance, Odometry, OdometryError, Quality, SampleFault,
    SensorSample,
};
use r2r::std_srvs::srv::Empty;
use r2r::{builtin_interfaces, nav_msgs, QosProfile};

const DEFAULT_MODEL: &str =
    "/opt/legged-neural-odometry/share/legged_neural_odometry/models/estimator.onnx";
const DEFAULT_JOINTS: [i64; 8] = [0, 1, 2, 3, 6, 7, 8, 9];

struct Config {
    lowstate_topic: String,
    odom_topic: String,
    odom_frame: String,
    base_frame: String,
    joint_index: [usize; 8],
    joint_sign: [f64; 8],
    model_path: String,
    yaw_drift_deg_min: Option<f64>,
}

struct RunState {
    odom: Odometry,
    last_quality: Option<Quality>,
    announced_ready: bool,
    wait_log_t: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = r2r::Context::create()?;
    let mut node = r2r::Node::create(ctx, "legged_neural_odometry", "")?;
    let config = read_config(&node)?;
    let runtime = onnxruntime_library().ok_or("libonnxruntime.so not found; set ORT_DYLIB_PATH")?;
    let odom = Odometry::open(
        config.model_path.as_ref(),
        &runtime,
        config.yaw_drift_deg_min,
    )?;
    eprintln!(
        "legged_neural_odometry: model {} runtime {}",
        config.model_path,
        runtime.display()
    );

    let state = Arc::new(Mutex::new(RunState {
        odom,
        last_quality: None,
        announced_ready: false,
        wait_log_t: f64::NEG_INFINITY,
    }));
    let reset_name = format!("{}/reset", node.fully_qualified_name()?);
    let mut reset =
        node.create_service::<Empty::Service>(&reset_name, QosProfile::services_default())?;
    let reset_state = Arc::clone(&state);
    let mut pool = LocalPool::new();
    pool.spawner().spawn_local(async move {
        while let Some(request) = reset.next().await {
            if let Ok(mut state) = reset_state.lock() {
                state.odom.reset();
                state.last_quality = None;
                state.announced_ready = false;
                state.wait_log_t = f64::NEG_INFINITY;
            }
            eprintln!("legged_neural_odometry: reset");
            let _ = request.respond(Empty::Response {});
        }
    })?;

    let mut sub = node.subscribe::<r2r::unitree_hg::msg::LowState>(
        &config.lowstate_topic,
        QosProfile::default(),
    )?;
    let mut odom_qos = QosProfile::sensor_data();
    odom_qos.depth = 10;
    let publisher =
        node.create_publisher::<nav_msgs::msg::Odometry>(&config.odom_topic, odom_qos)?;
    let clock = node.get_ros_clock();
    let sub_state = Arc::clone(&state);
    pool.spawner().spawn_local(async move {
        while let Some(message) = sub.next().await {
            let now = {
                let mut clock = clock.lock().unwrap_or_else(|poison| poison.into_inner());
                match clock.get_now() {
                    Ok(now) => now,
                    Err(error) => {
                        eprintln!("legged_neural_odometry: clock error: {error}");
                        continue;
                    }
                }
            };
            let stamp = r2r::Clock::to_builtin_time(&now);
            let t = now.as_secs_f64();
            let Some(sample) = sample_from_lowstate(&message, &config, t) else {
                continue;
            };
            let mut state = sub_state.lock().unwrap_or_else(|poison| poison.into_inner());
            match state.odom.update(&sample) {
                Ok(None) => {
                    if !state.announced_ready && t - state.wait_log_t >= 2.0 {
                        let norm = sample.accel[0].hypot(sample.accel[1]).hypot(sample.accel[2]);
                        eprintln!(
                            "legged_neural_odometry: waiting for 0.75 s of stillness and a 25-frame window; accel norm {norm:.3} (expect about 9.81)"
                        );
                        state.wait_log_t = t;
                    }
                }
                Ok(Some(result)) => {
                    if !state.announced_ready {
                        eprintln!("legged_neural_odometry: publishing {}", config.odom_topic);
                        state.announced_ready = true;
                    }
                    if state.last_quality != Some(result.quality) {
                        eprintln!("legged_neural_odometry: quality {}", result.quality.as_str());
                        state.last_quality = Some(result.quality);
                    }
                    if let Err(error) = publisher.publish(&odometry_message(&config, &stamp, &result)) {
                        eprintln!("legged_neural_odometry: publish failed: {error}");
                    }
                }
                Err(OdometryError::Sample(SampleFault::Yaw)) => {
                    eprintln!("legged_neural_odometry: ignoring sample with invalid IMU yaw");
                }
                Err(OdometryError::Sample(fault)) => {
                    eprintln!("legged_neural_odometry: {fault}; resetting");
                    state.odom.reset();
                    state.last_quality = None;
                    state.announced_ready = false;
                    state.wait_log_t = t;
                }
                Err(OdometryError::Model(message)) => {
                    eprintln!("legged_neural_odometry: inference failed: {message}");
                }
            }
        }
    })?;

    loop {
        node.spin_once(Duration::from_millis(2));
        pool.run_until_stalled();
    }
}

fn read_config(node: &r2r::Node) -> Result<Config, Box<dyn std::error::Error>> {
    let joint_index = node
        .get_parameter::<Option<Vec<i64>>>("joint_index")?
        .unwrap_or_else(|| DEFAULT_JOINTS.to_vec());
    if joint_index.len() != 8 || joint_index.iter().any(|index| *index < 0 || *index >= 35) {
        return Err("joint_index must be eight LowState motor indices in 0..35".into());
    }
    let mut joints = [0; 8];
    for (slot, index) in joints.iter_mut().zip(joint_index) {
        *slot = index as usize;
    }
    let signs = node
        .get_parameter::<Option<Vec<f64>>>("joint_sign")?
        .unwrap_or_else(|| vec![1.0; 8]);
    if signs.len() != 8 {
        return Err("joint_sign must have eight entries".into());
    }
    let mut joint_sign = [1.0; 8];
    for (slot, sign) in joint_sign.iter_mut().zip(signs) {
        *slot = if sign < 0.0 { -1.0 } else { 1.0 };
    }
    Ok(Config {
        lowstate_topic: node
            .get_parameter::<Option<String>>("lowstate_topic")?
            .unwrap_or_else(|| "/lowstate".into()),
        odom_topic: node
            .get_parameter::<Option<String>>("odom_topic")?
            .unwrap_or_else(|| "/g1/odom".into()),
        odom_frame: node
            .get_parameter::<Option<String>>("odom_frame")?
            .unwrap_or_else(|| "odom".into()),
        base_frame: node
            .get_parameter::<Option<String>>("base_frame")?
            .unwrap_or_else(|| "pelvis".into()),
        joint_index: joints,
        joint_sign,
        model_path: node
            .get_parameter::<Option<String>>("model_path")?
            .unwrap_or_else(|| DEFAULT_MODEL.into()),
        yaw_drift_deg_min: node.get_parameter("yaw_drift_deg_min")?,
    })
}

fn sample_from_lowstate(
    message: &r2r::unitree_hg::msg::LowState,
    config: &Config,
    t: f64,
) -> Option<SensorSample> {
    let gyro = three(&message.imu_state.gyroscope)?;
    let accel = three(&message.imu_state.accelerometer)?;
    let rpy = three(&message.imu_state.rpy)?;
    let needed = config.joint_index.iter().copied().max().unwrap_or(0);
    if message.motor_state.len() <= needed {
        eprintln!(
            "legged_neural_odometry: LowState has {} motors, need index {}",
            message.motor_state.len(),
            needed
        );
        return None;
    }
    let mut q = [0.0; 8];
    let mut dq = [0.0; 8];
    for i in 0..8 {
        let motor = &message.motor_state[config.joint_index[i]];
        q[i] = motor.q as f64 * config.joint_sign[i];
        dq[i] = motor.dq as f64 * config.joint_sign[i];
    }
    Some(SensorSample {
        t,
        gyro,
        accel,
        q,
        dq,
        imu_yaw: rpy[2],
        imu_yaw_t: t,
    })
}

fn three(values: &[f32]) -> Option<[f64; 3]> {
    if values.len() < 3 {
        return None;
    }
    Some([values[0] as f64, values[1] as f64, values[2] as f64])
}

fn odometry_message(
    config: &Config,
    stamp: &builtin_interfaces::msg::Time,
    result: &legged_neural_odometry::OdometryResult,
) -> nav_msgs::msg::Odometry {
    let (pose_cov, twist_cov) = ros_covariance(&result.pose_cov, &result.velocity_cov);
    let half = result.pose[2] * 0.5;
    nav_msgs::msg::Odometry {
        header: r2r::std_msgs::msg::Header {
            stamp: stamp.clone(),
            frame_id: config.odom_frame.clone(),
        },
        child_frame_id: config.base_frame.clone(),
        pose: r2r::geometry_msgs::msg::PoseWithCovariance {
            pose: r2r::geometry_msgs::msg::Pose {
                position: r2r::geometry_msgs::msg::Point {
                    x: result.pose[0],
                    y: result.pose[1],
                    z: 0.0,
                },
                orientation: r2r::geometry_msgs::msg::Quaternion {
                    x: 0.0,
                    y: 0.0,
                    z: half.sin(),
                    w: half.cos(),
                },
            },
            covariance: pose_cov.to_vec(),
        },
        twist: r2r::geometry_msgs::msg::TwistWithCovariance {
            twist: r2r::geometry_msgs::msg::Twist {
                linear: r2r::geometry_msgs::msg::Vector3 {
                    x: result.velocity[0],
                    y: result.velocity[1],
                    z: 0.0,
                },
                angular: r2r::geometry_msgs::msg::Vector3 {
                    x: 0.0,
                    y: 0.0,
                    z: result.gyro_z,
                },
            },
            covariance: twist_cov.to_vec(),
        },
    }
}
