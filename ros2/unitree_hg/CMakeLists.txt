cmake_minimum_required(VERSION 3.8)
project(unitree_hg)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/IMUState.msg"
  "msg/MotorState.msg"
  "msg/LowState.msg"
)

ament_package()
