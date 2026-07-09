# Code Conventions

## Python Style

**Linting**: Ruff (select: E, F, I, UP, B, C4, SIM; ignore: E203, E501)
**Formatting**: Black (line-length: 99)
**Type Hints**: MyPy (Python 3.10 target, warn_return_any, ignore_missing_imports)

**Naming**:
- Variables/functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`

**Imports**: ISort (profile: black, known_first_party: src)

## Lightning Patterns

**LightningModule** (`src/models/`):
- Network architecture passed via `net` parameter (separate from module)
- Lifecycle methods: `__init__`, `forward`, `training_step`, `validation_step`, `test_step`, `configure_optimizers`
- Metrics via TorchMetrics (Accuracy, MeanMetric, MaxMetric)
- Use `self.log()` for metric tracking

**LightningDataModule** (`src/datamodules/`):
- Required methods: `setup()`, `train_dataloader()`, `val_dataloader()`, `test_dataloader()`
- Dataset components in `src/datamodules/components/`

## Hydra Configuration

**Pattern**:
- `_target_`: Class path for instantiation
- `_partial_: true`: Defer instantiation (for optimizers/schedulers needing model parameters)
- Defaults list composition in main config

**Structure**:
- `configs/train.yaml`: Main config with defaults list
- `configs/model/`: Model + optimizer + scheduler
- `configs/datamodule/`: Dataset settings
- `configs/trainer/`: Trainer configs (cpu, gpu, ddp)
- `configs/callbacks/`, `configs/logger/`: Modular components
- `configs/experiment/`: Preset hyperparameter combinations

**CLI Overrides**: `python src/train.py <config_group>=<name> <param>=<value>`

## Utility Patterns

**Entry Point Setup**:
```python
import pyrootutils
root = pyrootutils.setup_root(__file__, pythonpath=True, cwd=True)
```

**Task Wrapper**:
```python
@task_wrapper
def train(cfg: DictConfig) -> tuple[dict, dict]:
    ...
```

**Checkpoint Loading**: Use `load_checkpoint()` from `src/utils/torch_utils.py` (supports key replacement, extra keys filtering)

## Distributed Training

**Configs**:
- `trainer=ddp`: Requires `torchrun` or Lightning DDP launch
- `trainer=ddp_spawn`: Works without torchrun (spawns processes)
- `trainer=ddp_no_unused`: DDP without `find_unused_parameters`

## Comment Philosophy

Minimize comments - code should be self-explanatory. Only add for:
- Complex logic explanation
- Non-obvious business rules
- Important constraints/limitations

Use English for comments and commit messages.