# Warmup Learning Rate Schedulers

**Location**: `src/schedulers/warmup.py`

**Base Class**: `torch.optim.lr_scheduler.LRScheduler` (PyTorch 2.0+)

## Implemented Schedulers

**WarmupCosineScheduler**:
- Linear warmup → Cosine annealing to `eta_min`
- Formula: `lr = eta_min + (base_lr - eta_min) * (1 + cos(progress * π)) / 2`

**WarmupLinearScheduler**:
- Linear warmup → Linear decay to `eta_min`
- Formula: `lr = eta_min + (base_lr - eta_min) * (1 - progress)`

**WarmupConstantScheduler**:
- Linear warmup → Constant LR
- After warmup: `lr = base_lr`

**Warmup Formula** (all schedulers):
```python
lr = base_lr * (step + 1) / warmup_steps
```

**Config Pattern**: Hydra `_partial_: true`
```yaml
_target_: src.schedulers.warmup.WarmupCosineScheduler
_partial_: true
warmup_steps: 10000
max_steps: 50000
eta_min: 0.0
```

**Current Issue** (See `mem:optimization_plan`):
- Line 59: `torch.tensor(progress * torch.pi)` without device specification
- Should use `math.pi` and specify device or avoid tensor creation

**Improvement Opportunities**:
- Add validation: `warmup_steps < max_steps`
- Add `verbose` parameter for logging
- Implement `_get_closed_form_lr` for better reproducibility