# Suggested Commands

## Development Commands

**Training**:
```bash
# CPU training
python src/train.py trainer=cpu

# Single GPU
python src/train.py trainer=gpu

# Multi-GPU DDP
python src/train.py trainer=ddp

# With experiment preset
python src/train.py experiment=example

# Override parameters
python src/train.py trainer.max_epochs=20 datamodule.batch_size=64
```

**Evaluation**:
```bash
python src/eval.py ckpt_path=<path_to_checkpoint>
```

**Testing**:
```bash
# Quick tests (skip slow)
pytest -k "not slow"

# All tests
pytest

# With coverage
pytest --cov=src --cov-report=html
```

**Code Quality**:
```bash
# Run all pre-commit hooks
pre-commit run -a

# Or via Makefile
make format
```

**Docker**:
```bash
docker-compose run train-cpu    # CPU training
docker-compose run train-dev    # Development mode
```

## System Commands (Darwin)

Standard Unix commands work as expected. No special Darwin-specific variants needed.

**Git**:
```bash
git status
git add -A
git commit -m "type(scope): description"
git push
```

**File Operations**:
```bash
ls -la
find . -name "*.py"
grep -r "pattern" src/
```

## Serena Memory Management

```bash
serena memories check  # Validate memory references
```