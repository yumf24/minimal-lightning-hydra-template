# Models Module Overview

**Location**: `src/models/`

**Main File**: `src/models/template_module.py`

**Key Class**: `MNISTLitModule` (LightningModule implementation)

**Implementation Pattern**:
- Network architecture passed via `net` parameter (separate from module)
- Uses `self.save_hyperparameters(logger=False, ignore=["net"])`
- Implements standard lifecycle: `__init__`, `forward`, `training_step`, `validation_step`, `test_step`, `configure_optimizers`
- Hook methods: `on_train_start`, `on_validation_epoch_end` (metric reset and best tracking)

**Metrics**:
- TorchMetrics: `Accuracy(task="multiclass", num_classes=10)`
- Aggregation: `MeanMetric()` for loss, `MaxMetric()` for best accuracy
- Per-split metrics: train_acc, val_acc, test_acc, train_loss, val_loss, test_loss

**Optimizer Configuration**:
- Uses Hydra `_partial_: true` pattern
- `configure_optimizers()` instantiates optimizer with `self.hparams.optimizer(params=self.parameters())`
- Optional scheduler via `self.hparams.scheduler(optimizer=optimizer)`

**Components**:
- `src/models/components/simple_dense_net.py`: Basic feedforward network with BatchNorm

**Current Issue** (See `mem:optimization_plan`):
- Uses deprecated import: `from pytorch_lightning import LightningModule`
- Should use: `from lightning.pytorch import LightningModule`

**Config Files**: `configs/model/*.yaml`