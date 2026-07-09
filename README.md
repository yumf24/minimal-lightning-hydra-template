<div align="center">

# Deep Learning Template

English | [中文](README_CN.md)

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://lightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/Lightning-2.0+-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-1.3+-89b8cd"></a>
<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.9+-3776ab?logo=python&logoColor=white"></a>
<a href="https://github.com/pre-commit/pre-commit"><img alt="Pre-commit" src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white"></a>

</div>

## Description

A production-ready deep learning project template featuring:

- **PyTorch Lightning 2.x** for structured training loops and distributed training
- **Hydra** for hierarchical configuration management
- **TorchMetrics** for standardized metric computation
- **Python 3.9+ modern syntax** - native type hints, modern logging practices
- **PyTorch 2.0+ security features** - safe checkpoint loading with weights_only
- Custom optimizers (Lion, DAdaptAdam)
- Warmup schedulers (Cosine, Linear, Constant)
- Docker and docker-compose support
- GitHub Actions CI/CD with comprehensive pre-commit hooks

## Installation

**Requirements**: Python 3.9 or higher (for modern type hint syntax)

### Using pip

```bash
# clone project
git clone https://github.com/your-username/your-repo-name
cd your-repo-name

# create virtual environment (optional)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# install package in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Using Docker

```bash
# build and run CPU training
docker-compose run train-cpu

# run with local code changes (development mode)
docker-compose run train-dev
```

## Quick Start

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU (single)
python src/train.py trainer=gpu

# train on multiple GPUs with DDP
python src/train.py trainer=ddp

# run with custom experiment config
python src/train.py experiment=example
```

## Project Structure

```
.
├── configs/              # Hydra configuration files
│   ├── model/            # Model configurations
│   ├── datamodule/       # Data module configurations
│   ├── trainer/          # Trainer configurations (cpu, gpu, ddp)
│   ├── callbacks/        # Callback configurations
│   ├── logger/           # Logger configurations (wandb, tensorboard, etc.)
│   └── experiment/       # Experiment presets
├── src/                  # Source code
│   ├── models/           # LightningModule implementations
│   │   └── components/   # Neural network components
│   ├── datamodules/      # LightningDataModule implementations
│   │   └── components/   # Dataset implementations
│   ├── optimizers/       # Custom optimizers (lion, dadapt_adam)
│   ├── schedulers/       # Custom LR schedulers (warmup)
│   └── utils/            # Utility functions
├── tests/                # pytest test suite
├── scripts/              # Helper scripts
├── .github/workflows/    # GitHub Actions CI
├── Dockerfile            # Container definition
├── docker-compose.yml    # Docker compose services
├── pyproject.toml        # Project metadata and tool configs
└── requirements.txt      # pip fallback dependencies
```

## Configuration

Override any parameter from command line:

```bash
# change number of epochs
python src/train.py trainer.max_epochs=20

# change batch size
python src/train.py datamodule.batch_size=64

# use different learning rate
python src/train.py model.optimizer.lr=0.0001

# combine multiple overrides
python src/train.py trainer.max_epochs=20 datamodule.batch_size=64 model.optimizer.lr=0.0001
```

## Testing

```bash
# run quick tests (skip slow tests)
pytest -k "not slow"

# run all tests
pytest

# run with coverage
pytest --cov=src --cov-report=html
```

## Custom Optimizers

This template includes custom optimizer implementations:

- **Lion** (`src/optimizers/lion.py`): Google's Lion optimizer with modern type hints
- **DAdaptAdam** (`src/optimizers/dadapt_adam.py`): Learning rate free optimizer with proper logging

```bash
python src/train.py model.optimizer._target_=src.optimizers.lion.Lion
```

## Warmup Schedulers

Use warmup schedulers for stable training. All schedulers use math.pi for device-agnostic computation:

```bash
python src/train.py model.scheduler._target_=src.schedulers.warmup.WarmupCosineScheduler \
    model.scheduler.warmup_steps=1000 model.scheduler.max_steps=10000
```

Available schedulers:
- `WarmupCosineScheduler`: Linear warmup + cosine annealing
- `WarmupLinearScheduler`: Linear warmup + linear decay
- `WarmupConstantScheduler`: Linear warmup + constant LR

## Distributed Training

```bash
# DDP with 4 GPUs
python src/train.py trainer=ddp

# DDP spawn (works without torchrun)
python src/train.py trainer=ddp_spawn
```

## Logging

Support for multiple experiment trackers:

```bash
# WandB
python src/train.py logger=wandb

# TensorBoard
python src/train.py logger=tensorboard

# Multiple loggers
python src/train.py logger=[wandb,tensorboard]
```

**Note**: All logging uses standard Python logging module (replaced print statements for better control).

## Modern Features

This template follows Python 3.9+ and PyTorch 2.0+ best practices:

### Type Hints
```python
# Native type hints (no typing imports needed)
def get_lr(self) -> list[float]:
def setup(self, stage: str | None = None):
callbacks: list[Callback] = []
```

### Security
```python
# Safe checkpoint loading (PyTorch 2.0+)
state_dict = torch.load(checkpoint_path, weights_only=True)
```

### Logging
```python
# Standardized logging throughout codebase
log.info(f"Instantiating model <{cfg.model._target_}>")
log.warning("Best ckpt not found!")
log.error(f"Failed to save video: {e}")
```

### Device-Agnostic Computation
```python
# Warmup schedulers use math.pi (no device mismatch)
import math
cos_value = math.cos(progress * math.pi)
```

## License

MIT License