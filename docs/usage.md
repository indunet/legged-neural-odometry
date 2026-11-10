# 使用方法

## 推理接口

```python
from g1_odom.runtime import Odometry
from g1_odom.sensors import SensorSample

odom = Odometry()            # 默认 releases/estimator.onnx；也可传成员列表 + calibration='releases/calibration.json'
# yaw 是硬件 IMU 输出、已变换到骨盆坐标系的姿态 yaw，单位 rad，例如 low_state.imu_state.rpy[2]（需核对安装坐标系）。
result = odom.update(SensorSample(t, gyro, accel, q, dq, imu_yaw=yaw, imu_yaw_t=t))
# result: None（尚未就绪／未形成新帧）或
# {t, velocity: [vx, vy], pose: [x, y, yaw], stationary, ready, latency_us,
#  velocity_cov,   # [vx, vy] 2x2 行优先
#  pose_cov,       # [x, y, yaw] 3x3 行优先，对应返回的 pose，可直接映射到 ROS 协方差
#  uncertainty,    # {'aleatoric_std', 'epistemic_std', 'epistemic_speed_std'}
#  quality}        # 'ok' | 'warn' | 'alarm'：1 秒平滑的种子间分歧对比验证集 P99 / P99.9
```

## 输入约定

- 时间戳为单调递增秒；IMU 单位 rad/s、m/s²，竖直静止时比力约为 `[0, 0, +9.81]`。传感器坐标 x 前、y 左、z 上。
- q/dq 是八个髋膝关节，单位 rad、rad/s；顺序见 `g1_odom.common.JOINTS`（左右腿依次 hip_pitch、hip_roll、hip_yaw、knee）。
- 传感器 500 Hz，网络特征 50 Hz。归一化包含在 ONNX 内，特征名、采样率、窗口、标签坐标系与校准参数写入模型元数据。
- 模拟 IMU 位于骨盆原点，姿态与骨盆一致；接真实 IMU 时需按外参变换，有位置偏置时要处理杆臂效应。
- 速度位于重力对齐的骨盆航向坐标系；位姿相对启动就绪时刻，yaw 以首次就绪时为零点，连续处理 ±π 跨界。
- `imu_yaw` 缺失或非法时报错，不会退回陀螺积分；`imu_yaw_t` 不能晚于样本，也不能比样本旧超过 20 ms（这是拒收上限，不是推荐延迟）。
- 网络重力特征由原始陀螺与加速度构造，需要至少 0.75 秒连续静止完成标定；仿真开头用固定支架提供 1 秒静止。

## 不确定度与使用限制

- `yaw_drift_deg_min` 默认 1°/min，接真机时应换成 IMU 实测值。
- 协方差只在训练分布内校准过；`quality` 为 warn/alarm 时，融合端应降低里程计权重或只把它当作告警。
- 跌倒后应停止或重置里程计。

## 发布文件

`releases/` 下：

- `estimator.onnx`：三种子集成，输入 `features`（1 × 25 × 25），输出 `velocity`、`aleatoric_var`、`epistemic_var`；元数据 `calibration` 含校准参数。
- `members/estimator_seed{0,1,2}.onnx`：单个种子，输出 `velocity`、`log_var`。
- `calibration.json`、`model.json`（窗口与均值来源的验证集依据、标签／训练器／训练数据哈希）、`runtime.json`（输出字段约定）。
- `implementation.json`：发布时的实现文件哈希快照，路径为目录整理前的布局（`g1_odom/…`、`web/…`），仅作历史记录。

## Web 控制台

```bash
scripts/run_web.sh               # http://127.0.0.1:18078，可指定 --host 0.0.0.0 --port 18078
```

页面分两页：

- 「里程计」页：整个页面背景就是实时 G1 平地仿真画面（MuJoCo 离屏渲染，1280×720、30 帧/秒，以 MJPEG 视频流 `/api/stream` 推送）；
  速度指令、轨迹、速度曲线、水平误差与 2σ 曲线、位置 1σ 和估计质量以半透明面板悬浮在画面上。右上角按钮或按 `H` 键可隐藏／显示面板。
- 「训练与报告」页：数据、流程阶段与发布状态，可启动／停止流程，并显示精度、不确定度覆盖率和在线回归结果。

平地步行使用 Unitree 官方 Mjlab G1 velocity 策略，它只驱动仿真，与里程计网络无关。服务默认只监听本机，没有用户鉴权；
如需远程访问，可用 SSH 端口转发或指定监听地址。
