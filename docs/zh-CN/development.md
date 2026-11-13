# 开发与复现

## 环境

```bash
bash scripts/bootstrap-cpu.sh    # Python 3.12、uv、Linux；CPU 版 PyTorch，不需要 NVIDIA 驱动
sudo apt install libosmesa6      # MuJoCo 离屏渲染（无 GPU 时）
.venv/bin/python -m g1_odom fetch --verify
```

有 NVIDIA GPU 时可以用 `bash scripts/bootstrap.sh`，它按 `requirements-lock.txt` 安装 CUDA 版 PyTorch。
两个脚本都以可编辑模式安装本包（`pip install -e .`），并校验 `assets/unitree_g1/`。
项目使用 `src/` 布局，子进程通过 `python -m g1_odom...` 启动，因此需要先完成可编辑安装。
`pyproject.toml` 中的可选依赖：`sim`（MuJoCo 仿真）、`train`（PyTorch/ONNX 训练）、`dev`（测试与浏览器冒烟测试）。

G1 模型和步行策略放在 `assets/unitree_g1/`，来源与校验和记录在 `configs/assets.lock.json`。
`scripts/fetch_assets.py` 只做校验，不再克隆外部仓库。仿真用普通 MuJoCo 加载这些文件，不安装 mjlab。

本地资源配置：复制 `configs/resources.example.json` 为 `configs/resources.json`（Git 忽略）。
默认 `device` 为 `cpu`，`render_gl` 为 `osmesa`。有 GPU 时可把 `render_gl` 设为 `egl` 并设置 `render_gpu`。
`render_width`、`render_height`、`render_fps` 控制 Web 实时画面；已经设置的 `MUJOCO_GL` 环境变量优先于配置文件。

## 复现完整流程

```bash
# 采集（已有且配置一致的轨迹直接复用）→ 数据审计 → 标签审计 → 训练 → 校准 → 测试；每个阶段都可中断后恢复
.venv/bin/python -m g1_odom run --device cpu
# 单独执行某个阶段
.venv/bin/python -m g1_odom run --stage evaluate --skip-online
```

阶段为 `collect`、`audit`、`labels`、`train`、`calibrate`、`evaluate`。数据在 `data/g1_planar/`，检查点与日志在 `runs/`，
报告在 `reports/`。训练只接受 `data/g1_planar/` 下的数据；检查点记录标签、训练器与训练数据的哈希，不一致时拒绝复用，需换新的运行目录。
采样默认最多 8 个进程，可在 `resources.json` 的 `workers` 或命令行 `--workers` 里调小。
在 GPU 上完整重训约 15 分钟；同一流程在 CPU 上会明显更久，数值也不会和 GPU 训练逐位一致。

> `src/g1_odom/labels.py` 与 `trainer.py` 的内容哈希写入了检查点和 `releases/model.json`。修改这两个文件（包括仅调整格式）
> 会使现有检查点失效。

## 测试

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest
PLAYWRIGHT_BROWSERS_PATH="$PWD/.cache/browsers" .venv/bin/python scripts/web_smoke.py
```

浏览器冒烟测试需要 Playwright 及其 Chromium：`PLAYWRIGHT_BROWSERS_PATH="$PWD/.cache/browsers" .venv/bin/playwright install chromium`。
截图写入 `docs/images/`，结果写入 `reports/web_smoke.json`。
部分测试需要 `assets/unitree_g1/`（`fetch --verify`）或 `data/g1_planar/`，缺少数据的测试会自动跳过。

## Git

训练数据、依赖、检查点和运行文件都被 Git 忽略；`assets/unitree_g1/`、发布的 ONNX、源码、配置、报告可以纳入 Git。
