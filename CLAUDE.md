# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A deep learning project template using PyTorch Lightning 2.x and Hydra for configuration management. Supports distributed training (DDP), custom optimizers (Lion, DAdaptAdam), warmup schedulers, and multiple experiment loggers (WandB, TensorBoard, MLflow, etc.).

## Commands

### Training
```bash
# CPU training
python src/train.py trainer=cpu

# Single GPU
python src/train.py trainer=gpu

# Multi-GPU DDP
python src/train.py trainer=ddp

# With experiment config
python src/train.py experiment=example

# Override specific parameters
python src/train.py trainer.max_epochs=20 datamodule.batch_size=64 model.optimizer.lr=0.0001
```

### Evaluation
```bash
python src/eval.py ckpt_path=<path_to_checkpoint>
```

### Testing
```bash
# Quick tests (skip slow)
pytest -k "not slow"

# All tests
pytest

# With coverage
pytest --cov=src --cov-report=html
```

### Linting/Formatting
```bash
# Run pre-commit hooks
pre-commit run -a

# Or use Makefile
make format
```

### Docker
```bash
docker-compose run train-cpu    # CPU training
docker-compose run train-dev    # Development mode with local code
```

## Architecture

### Entry Points
- `src/train.py`: Main training script with Hydra config
- `src/eval.py`: Evaluation script for checkpoints

Both use `pyrootutils.setup_root()` at the top to:
- Add project root to PYTHONPATH (run from anywhere without package install)
- Set PROJECT_ROOT env variable for path configs
- Load .env file if present

### LightningModule Pattern (`src/models/`)
- `MNISTLitModule` as reference implementation
- Key sections: `__init__`, `forward`, `training_step`, `validation_step`, `test_step`, `configure_optimizers`
- Network architecture passed via `net` parameter (separate from module)
- Uses TorchMetrics for metric computation (Accuracy, MeanMetric, MaxMetric)

### LightningDataModule Pattern (`src/datamodules/`)
- `setup()` loads datasets
- `train_dataloader()`, `val_dataloader()`, `test_dataloader()` return DataLoader instances
- Dataset components in `src/datamodules/components/`

### Hydra Configuration (`configs/`)
Hierarchical config structure with composition:
- `train.yaml`: Main config with defaults list
- `model/`: Model architecture + optimizer + scheduler
- `datamodule/`: Dataset and dataloader settings
- `trainer/`: Trainer configs (cpu, gpu, ddp, ddp_spawn)
- `callbacks/`: Training callbacks (checkpoint, early stopping, LR monitor)
- `logger/`: Experiment loggers (wandb, tensorboard, etc.)
- `experiment/`: Preset hyperparameter combinations

Key config patterns:
- `_target_`: Class to instantiate via Hydra
- `_partial_: true`: Defer instantiation (for optimizers/schedulers needing model parameters)
- CLI overrides: `python src/train.py <config_group>=<name> <param>=<value>`

### Custom Optimizers (`src/optimizers/`)
- `lion.py`: Google's Lion optimizer
- `dadapt_adam.py`: Learning-rate-free optimizer
- Use via: `model.optimizer._target_=src.optimizers.lion.Lion`

### Warmup Schedulers (`src/schedulers/warmup.py`)
- `WarmupCosineScheduler`: Linear warmup + cosine annealing
- `WarmupLinearScheduler`: Linear warmup + linear decay
- `WarmupConstantScheduler`: Linear warmup + constant LR

## Utilities (`src/utils/`)
- `utils.py`: `task_wrapper` decorator, callback/logger instantiation, checkpoint utilities
- `pylogger.py`: Rank-zero-only logging
- `torch_utils.py`: `load_checkpoint` for flexible checkpoint loading
- `rich_utils.py`: Config printing and tag enforcement

## Adding New Components

### New Model
1. Create `src/models/<name>_module.py` extending LightningModule
2. Create network in `src/models/components/`
3. Add config `configs/model/<name>.yaml` with `_target_` pointing to module

### New Dataset
1. Create dataset class in `src/datamodules/components/`
2. Create DataModule in `src/datamodules/<name>_datamodule.py`
3. Add config `configs/datamodule/<name>.yaml`

## Distributed Training Notes
- `trainer=ddp`: Requires `torchrun` or Lightning's DDP launch
- `trainer=ddp_spawn`: Works without torchrun, spawns processes
- `trainer=ddp_no_unused`: DDP without `find_unused_parameters`