cmake_minimum_required(VERSION 3.8)
project(legged_neural_odometry)

find_package(ament_cmake REQUIRED)
find_package(rcl REQUIRED)
find_package(unitree_hg REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_srvs REQUIRED)

set(CARGO_TARGET_DIR "${CMAKE_CURRENT_BINARY_DIR}/cargo")

add_custom_target(${PROJECT_NAME}_cargo ALL
  COMMAND ${CMAKE_COMMAND} -E env "CARGO_TARGET_DIR=${CARGO_TARGET_DIR}"
          cargo build --release --locked --features ros
          --manifest-path "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  COMMENT "Building legged_neural_odometry"
  VERBATIM
)

install(PROGRAMS "${CARGO_TARGET_DIR}/release/${PROJECT_NAME}"
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/../../releases/estimator.onnx"
  DESTINATION share/${PROJECT_NAME}/models
)

ament_package()
