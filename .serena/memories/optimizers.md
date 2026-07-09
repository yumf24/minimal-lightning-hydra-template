# Custom Optimizers

**Location**: `src/optimizers/`

## Lion Optimizer (`lion.py`)

**Implementation**: Google Research's Lion optimizer (2023)
- Sign-based gradient update
- Default lr: 1e-4, betas: (0.9, 0.99)
- Optional Triton kernel (`lion_triton.py`) for GPU acceleration

**Config**: `configs/model/optimizer/lion.yaml`
```yaml
_target_: src.optimizers.lion.Lion
_partial_: true
lr: 1.e-4
betas: [0.9, 0.99]
weight_decay: 0.0
use_triton: False
```

## DAdaptAdam Optimizer (`dadapt_adam.py`)

**Implementation**: Meta's D-Adaptation Adam
- Self-adaptive learning rate (default lr=1.0)
- Adaptive D parameter estimation
- Optional decoupled weight decay (AdamW style)

**Config**: `configs/model/optimizer/dadapt_adam.yaml`
```yaml
_target_: src.optimizers.dadapt_adam.DAdaptAdam
_partial_: true
lr: 1.0
betas: [0.9, 0.999]
decouple: False
d0: 1.e-6
```

**Current Issue** (See `mem:optimization_plan`):
- Uses `print()` for logging (line 68)
- Should use Python logging module

**Usage Pattern**: Instantiated via Hydra with `_partial_: true`, passed to LightningModule's `configure_optimizers()`