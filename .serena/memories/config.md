# Configuration System

**Location**: `configs/`

**Framework**: Hydra (version 1.3.4)

## Main Config: `configs/train.yaml`

**Defaults Composition**:
```yaml
defaults:
  - _self_
  - datamodule: default
  - model: default
  - callbacks: default
  - logger: null
  - trainer: default
  - paths: default
  - extras: default
  - hydra: default
  - experiment: null
  - hparams_search: null
  - optional local: default
  - debug: null
```

**Override Order**: Later configs override earlier ones; `_self_` position controls current file priority.

## Key Patterns

**_target_**: Class path for Hydra instantiation
```yaml
_target_: src.models.template_module.MNISTLitModule
```

**_partial_: true**: Deferred instantiation (pass parameters later)
```yaml
optimizer:
  _target_: torch.optim.Adam
  _partial_: true
  lr: 0.001
```

**Used in**: Optimizers, schedulers (need model params), callbacks, loggers

## Config Groups

**model/**: Architecture + optimizer + scheduler
- Nested defaults: `optimizer: adamw`, `scheduler: none`

**datamodule/**: Dataset + DataLoader settings
- `mnist.yaml`, `default.yaml`

**trainer/**: Trainer configurations
- `cpu.yaml`, `gpu.yaml`, `ddp.yaml`, `ddp_spawn.yaml`

**callbacks/**: Training callbacks
- Nested defaults: model_summary, rich_progress_bar, learning_rate_monitor, model_checkpoint

**logger/**: Experiment loggers
- wandb, tensorboard, mlflow, neptune, comet, csv, many_loggers

**experiment/**: Preset hyperparameter combinations
- Uses `override /datamodule: mnist` syntax

## Path References

**Environment Variables** (`configs/paths/default.yaml`):
```yaml
root_dir: ${oc.env:PROJECT_ROOT}
data_dir: ${oc.env:DATA_ROOT,${paths.root_dir}/data/}
output_dir: ${hydra:runtime.output_dir}
```

## CLI Usage

```bash
python src/train.py <config_group>=<name> <param>=<value>
python src/train.py experiment=example trainer.max_epochs=20
```

## Instantiation in Code

```python
import hydra
datamodule = hydra.utils.instantiate(cfg.datamodule)
model = hydra.utils.instantiate(cfg.model)
trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
```