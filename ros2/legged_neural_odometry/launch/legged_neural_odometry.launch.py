import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("legged_neural_odometry")
    model = os.path.join(share, "models", "estimator.onnx")
    return LaunchDescription([
        Node(
            package="legged_neural_odometry",
            executable="legged_neural_odometry",
            name="legged_neural_odometry",
            output="screen",
            parameters=[{
                "model_path": model,
                "lowstate_topic": "/lowstate",
                "odom_topic": "/g1/odom",
                "odom_frame": "odom",
                "base_frame": "pelvis",
                "joint_index": [0, 1, 2, 3, 6, 7, 8, 9],
                "joint_sign": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "yaw_drift_deg_min": 1.0,
            }],
        ),
    ])
