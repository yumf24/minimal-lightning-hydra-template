# Utility Functions

**Location**: `src/utils/`

## Key Files

### `utils.py`
- **`task_wrapper` decorator**: Error handling + timing, saves `exec_time.log`
- **Checkpoint utilities**: `get_latest_checkpoint()`, `get_config()`, `configure_cfg_from_checkpoint()`
- **Logger instantiation**: `instantiate_loggers()` (rank_zero_only), `instantiate_callbacks()`
- **Hyperparameter logging**: `log_hyperparameters()`
- **WandB helpers**: `unflatten_wandb_config()`

### `torch_utils.py`
- **`load_checkpoint()`**: Flexible checkpoint loading with key replacement
- Supports `replace` tuple, `allow_extra_keys`, `extra_key` options

### `pylogger.py`
- **`get_pylogger()`**: Creates rank-zero-only logger for distributed training

### `rich_utils.py`
- **`print_config_tree()`**: Pretty print DictConfig as tree
- **`enforce_tags()`**: Interactive tag input for runs

## Current Issues (See `mem:optimization_plan`)

**torch_utils.py**:
- Line 22: Missing `weights_only=True` in `torch.load()`
- Security risk: allows arbitrary pickle execution

**utils.py**:
- Uses `os.path` instead of `pathlib.Path` (can modernize)
- Some type hints use old syntax (can use Python 3.9+ syntax)

## Patterns

**Entry Point Setup** (used in train.py, eval.py):
```python
import pyrootutils
root = pyrootutils.setup_root(__file__, pythonpath=True)
```

**Task Wrapper Pattern**:
```python
@task_wrapper
def train(cfg: DictConfig) -> tuple[dict, dict]:
    ...
```