# Technology Stack

**Language**: Python 3.9+ (tested on 3.11.15)

**Core Frameworks**:
- **PyTorch Lightning**: 2.6.5 (min: 2.0.0, max: <3.0.0)
- **PyTorch**: 2.12.1 (min: 2.0.0)
- **TorchVision**: 0.27.1 (min: 0.15.0)
- **TorchMetrics**: 1.9.0 (min: 1.0.0)

**Configuration Management**:
- **Hydra**: 1.3.4 (min: 1.3.0)
- **hydra-colorlog**: 1.3.0
- **hydra-optuna-sweeper**: 1.3.0 (hyperparameter search)

**Experiment Logging** (optional):
- WandB, MLflow, Neptune, Comet, TensorBoard

**Development Tools**:
- **pytest**: 7.0.0+ (testing)
- **pytest-cov**: 4.0.0+ (coverage)
- **pre-commit**: 3.0.0+ (git hooks)
- **ruff**: Linter/formatter (line-length: 99, target: py39)
- **black**: Formatter (line-length: 99)
- **mypy**: Type checker (Python 3.10 target)

**Utilities**:
- **pyrootutils**: 1.0.0+ (path setup)
- **rich**: 13.0.0+ (pretty printing)
- **einops**: 0.6.0+ (tensor operations)

**Build System**: setuptools (pyproject.toml)

**Version Pin Strategy**: Lower bounds only for core deps, upper bound for Lightning (<3.0.0)