# 开发与复现

## 环境

```bash
bash scripts/bootstrap.sh        # Python 3.12、uv、Linux 与匹配锁定 PyTorch CUDA 版本的 NVIDIA 驱动
.venv/bin/python -m g1_odom fetch --verify
```

`bootstrap.sh` 按 `requirements-lock.txt` 安装依赖，以可编辑模式安装本包（`pip install -e .`），并获取第三方资源。
项目使用 `src/` 布局，子进程通过 `python -m g1_odom...` 启动，因此需要先完成可编辑安装。
`pyproject.toml` 中的可选依赖：`sim`（MuJoCo 仿真）、`train`（PyTorch/ONNX 训练）、`dev`（测试与浏览器冒烟测试）。

资源脚本把固定版本的 Unitree RL Mjlab 放在 `third_party/`，来源与校验和记录在 `configs/assets.lock.json`。

本地资源配置：复制 `configs/resources.example.json` 为 `configs/resources.json`（Git 忽略），设置训练 GPU、渲染 GPU 和采样进程数。

## 复现完整流程

```bash
# 采集（已有且配置一致的轨迹直接复用）→ 数据审计 → 标签审计 → 训练 → 校准 → 测试；每个阶段都可中断后恢复
.venv/bin/python -m g1_odom run --device cuda:0
# 单独执行某个阶段
.venv/bin/python -m g1_odom run --stage evaluate --skip-online
```

阶段为 `collect`、`audit`、`labels`、`train`、`calibrate`、`evaluate`。数据在 `data/g1_planar/`，检查点与日志在 `runs/`，
报告在 `reports/`。训练只接受 `data/g1_planar/` 下的数据；检查点记录标签、训练器与训练数据的哈希，不一致时拒绝复用，需换新的运行目录。
一次使用一张 GPU、最多 8 个采样进程；完整重训约 15 分钟。

> `src/g1_odom/labels.py` 与 `trainer.py` 的内容哈希写入了检查点和 `releases/model.json`。修改这两个文件（包括仅调整格式）
> 会使现有检查点失效。

## 测试

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest
PLAYWRIGHT_BROWSERS_PATH="$PWD/.cache/browsers" .venv/bin/python scripts/web_smoke.py
```

浏览器冒烟测试需要 Playwright 及其 Chromium：`PLAYWRIGHT_BROWSERS_PATH="$PWD/.cache/browsers" .venv/bin/playwright install chromium`。
截图写入 `docs/images/`，结果写入 `reports/web_smoke.json`。
部分测试需要 `third_party/` 资源（`fetch`）或 `data/g1_planar/`，缺少数据的测试会自动跳过。

## Git

训练数据、依赖、下载资源、检查点和运行文件都被 Git 忽略；发布的 ONNX、源码、配置、报告可以纳入 Git。
