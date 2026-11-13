# Contributing

感谢你的关注！

## 开发环境

```bash
bash scripts/bootstrap.sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest
```

详见 [docs/development.md](docs/development.md)。

## 提交修改

- 提交前运行 `pytest`；改动 Web 页面时同时运行 `scripts/web_smoke.py`。
- 推理路径（`runtime.py`、`sensors.py`、`uncertainty.py`、`standstill.py`）不得依赖 `torch` 或 `mujoco`，测试会检查这一点。
- 不要修改 `labels.py`、`trainer.py`，除非准备重新训练并重新发布模型：它们的哈希写入了检查点和发布元数据。
- 不要提交数据、检查点（`*.pt`）和日志。`assets/unitree_g1/` 里的模型和策略随仓库提供；发布的里程计 ONNX 放在 `releases/`。
- 修改模型、标签或评测方式后，需要重新运行完整流程并更新 `releases/` 与 `reports/`。
- 提交贡献即表示你同意以 [Apache License 2.0](LICENSE) 授权你的贡献。
- 报告问题时请附上 Python、PyTorch、MuJoCo 版本和复现命令。
