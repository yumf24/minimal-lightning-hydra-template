# Testing Infrastructure

**Location**: `tests/`

**Framework**: pytest (version 7.0.0+)

## Test Files

**test_mnist_datamodule.py**: DataModule tests
- Parameterized: batch sizes [32, 128]
- Tests: prepare_data, setup, dataloaders, data integrity

**test_configs.py**: Configuration instantiation tests
- Validates all configs can be instantiated
- Tests Hydra composition

**test_train.py**: Training flow tests
- **fast_dev_run**: Quick validation
- **GPU/AMP tests**: Conditional with `@RunIf(min_gpus=1)`
- **DDP simulation**: Distributed training
- **Checkpoint resume**: Training recovery

**test_sweeps.py**: Hyperparameter search tests
- Hydra multirun
- Optuna integration

**test_eval.py**: Evaluation tests

## Fixtures (`conftest.py`)

**Two-level fixture strategy**:
```python
@pytest.fixture(scope="package")
def cfg_train_global() -> DictConfig:
    # Shared across tests

@pytest.fixture(scope="function")
def cfg_train(cfg_train_global, tmp_path) -> DictConfig:
    # Independent per test, uses temp path
```

## Conditional Test Execution

**Custom decorators** (`tests/helpers/run_if.py`):
```python
@RunIf(min_gpus=1)  # Skip if no GPU
@RunIf(wandb=True)  # Skip if wandb not installed
@RunIf(sh=True)     # Run shell commands
@pytest.mark.slow   # Mark slow tests
```

## pytest Configuration (`pyproject.toml`)

```toml
[tool.pytest.ini_options]
addopts = ["--color=yes", "--strict-markers", "--doctest-modules"]
filterwarnings = ["ignore::DeprecationWarning", "ignore::UserWarning"]
markers = ["slow: slow tests"]
minversion = "6.0"
testpaths = "tests/"
```

## Coverage Configuration

```toml
[tool.coverage.report]
exclude_lines = [
    "pragma: nocover",
    "raise NotImplementedError",
    "if __name__ == .__main__.:"
]
```

## Test Commands

```bash
# Quick (skip slow)
pytest -k "not slow"

# Full suite
pytest

# Coverage
pytest --cov=src --cov-report=html

# Conditional
pytest -m "not slow"  # Skip slow tests
pytest -k "gpu"       # Run GPU-related tests only
```

## Test Pattern

- **Instantiation test**: Verify config creates objects
- **Functional test**: Verify behavior (e.g., dataloader works)
- **Integration test**: Verify full flow (e.g., training loop)