<div align="center">

# Deep Learning Template 深度学习模板

[English](README.md) | 中文

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://lightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/Lightning-2.0+-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-1.3+-89b8cd"></a>
<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.9+-3776ab?logo=python&logoColor=white"></a>
<a href="https://github.com/pre-commit/pre-commit"><img alt="Pre-commit" src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white"></a>

</div>

## 简介

一个生产级深度学习项目模板，包含以下特性：

- **PyTorch Lightning 2.x** - 结构化训练循环和分布式训练
- **Hydra** - 分层配置管理
- **TorchMetrics** - 标准化指标计算
- **Python 3.9+ 现代语法** - 原生类型提示，现代化日志实践
- **PyTorch 2.0+ 安全特性** - 使用 weights_only 安全加载检查点
- 自定义优化器（Lion, DAdaptAdam）
- 预热学习率调度器（余弦、线性、恒定）
- Docker 和 docker-compose 支持
- GitHub Actions CI/CD，配备全面的 pre-commit hooks

## 安装

**要求**：Python 3.9 或更高版本（支持现代类型提示语法）

### 使用 pip

```bash
# 克隆项目
git clone https://github.com/your-username/your-repo-name
cd your-repo-name

# 创建虚拟环境（可选）
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# 以可编辑模式安装包及开发依赖
pip install -e ".[dev]"
```

### 使用 Docker

```bash
# 构建并运行 CPU 训练
docker-compose run train-cpu

# 使用本地代码更改运行（开发模式）
docker-compose run train-dev
```

## 快速开始

```bash
# CPU 训练
python src/train.py trainer=cpu

# 单 GPU 训练
python src/train.py trainer=gpu

# 多 GPU DDP 训练
python src/train.py trainer=ddp

# 使用自定义实验配置
python src/train.py experiment=example
```

## 项目结构

```
.
├── configs/              # Hydra 配置文件
│   ├── model/            # 模型配置
│   ├── datamodule/       # 数据模块配置
│   ├── trainer/          # Trainer 配置（cpu, gpu, ddp）
│   ├── callbacks/        # 回调配置
│   ├── logger/           # 日志器配置（wandb, tensorboard 等）
│   └── experiment/       # 实验预设
├── src/                  # 源代码
│   ├── models/           # LightningModule 实现
│   │   └── components/   # 神经网络组件
│   ├── datamodules/      # LightningDataModule 实现
│   │   └── components/   # 数据集实现
│   ├── optimizers/       # 自定义优化器（lion, dadapt_adam）
│   ├── schedulers/       # 自定义学习率调度器（warmup）
│   └── utils/            # 工具函数
├── tests/                # pytest 测试套件
├── scripts/              # 辅助脚本
├── .github/workflows/    # GitHub Actions CI
├── Dockerfile            # 容器定义
├── docker-compose.yml    # Docker compose 服务
├── pyproject.toml        # 项目元数据和工具配置
└── requirements.txt      # pip 依赖回退
```

## 配置

从命令行覆盖任意参数：

```bash
# 改变训练轮数
python src/train.py trainer.max_epochs=20

# 改变批次大小
python src/train.py datamodule.batch_size=64

# 使用不同学习率
python src/train.py model.optimizer.lr=0.0001

# 组合多个覆盖参数
python src/train.py trainer.max_epochs=20 datamodule.batch_size=64 model.optimizer.lr=0.0001
```

## 测试

```bash
# 运行快速测试（跳过慢速测试）
pytest -k "not slow"

# 运行所有测试
pytest

# 运行带覆盖率报告
pytest --cov=src --cov-report=html
```

## 自定义优化器

本模板包含自定义优化器实现：

- **Lion** (`src/optimizers/lion.py`): Google 的 Lion 优化器，配备现代类型提示
- **DAdaptAdam** (`src/optimizers/dadapt_adam.py`): 无学习率优化器，配备规范化日志

```bash
python src/train.py model.optimizer._target_=src.optimizers.lion.Lion
```

## 预热学习率调度器

使用预热调度器实现稳定训练。所有调度器使用 math.pi 实现设备无关计算：

```bash
python src/train.py model.scheduler._target_=src.schedulers.warmup.WarmupCosineScheduler \
    model.scheduler.warmup_steps=1000 model.scheduler.max_steps=10000
```

可用调度器：
- `WarmupCosineScheduler`: 线性预热 + 余弦退火
- `WarmupLinearScheduler`: 线性预热 + 线性衰减
- `WarmupConstantScheduler`: 线性预热 + 恒定学习率

## 分布式训练

```bash
# 4 GPU DDP 训练
python src/train.py trainer=ddp

# DDP spawn（无需 torchrun）
python src/train.py trainer=ddp_spawn
```

## 日志记录

支持多种实验追踪器：

```bash
# WandB
python src/train.py logger=wandb

# TensorBoard
python src/train.py logger=tensorboard

# 多日志器
python src/train.py logger=[wandb,tensorboard]
```

**注意**：所有日志使用标准 Python logging 模块（已替换 print 语句以获得更好的控制）。

## 现代特性

本模板遵循 Python 3.9+ 和 PyTorch 2.0+ 最佳实践：

### 类型提示
```python
# 原生类型提示（无需 typing 导入）
def get_lr(self) -> list[float]:
def setup(self, stage: str | None = None):
callbacks: list[Callback] = []
```

### 安全性
```python
# 安全的检查点加载（PyTorch 2.0+）
state_dict = torch.load(checkpoint_path, weights_only=True)
```

### 日志记录
```python
# 整个代码库标准化日志
log.info(f"Instantiating model <{cfg.model._target_}>")
log.warning("Best ckpt not found!")
log.error(f"Failed to save video: {e}")
```

### 设备无关计算
```python
# 预热调度器使用 math.pi（无设备不匹配问题）
import math
cos_value = math.cos(progress * math.pi)
```

## 许可证

MIT License