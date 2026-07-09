# Core Project Structure

**Project**: PyTorch Lightning template with Hydra configuration

**Entry Points**:
- `src/train.py`: Main training script with Hydra CLI
- `src/eval.py`: Checkpoint evaluation script

**Module Organization**:
- `src/models/`: LightningModule implementations, see `mem:models/core`
- `src/datamodules/`: LightningDataModule + dataset components, see `mem:data/core`
- `src/optimizers/`: Custom optimizers (Lion, DAdaptAdam), see `mem:optimizers`
- `src/schedulers/`: Warmup schedulers, see `mem:schedulers`
- `src/utils/`: Training utilities (task_wrapper, checkpoint loading), see `mem:utils`

**Configuration System**:
- `configs/`: Hydra hierarchical configs, see `mem:config`
- Composition via `_target_` + `_partial_: true` pattern

**Testing**: `tests/` with pytest, see `mem:testing`

**Key Patterns**:
- `pyrootutils.setup_root()` at entry points for PYTHONPATH setup
- `task_wrapper` decorator for error handling and timing
- Hydra instantiation with `_partial_: true` for deferred optimizer/scheduler creation

**Current Versions** (as of 2026-07-09):
- PyTorch Lightning: 2.6.5
- PyTorch: 2.12.1
- TorchMetrics: 1.9.0
- Hydra: 1.3.4
- Python: 3.11.15