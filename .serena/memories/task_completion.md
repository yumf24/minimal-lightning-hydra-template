# Task Completion Checklist

When completing a coding task, run the following in order:

## 1. Code Quality Checks

```bash
# Run all pre-commit hooks (includes ruff, black, isort)
pre-commit run -a
```

Or individually:
```bash
# Format with black
black src/ tests/

# Lint with ruff
ruff check src/ tests/ --fix

# Type check with mypy (optional)
mypy src/
```

## 2. Run Tests

```bash
# Quick test (skip slow tests)
pytest -k "not slow"

# Full test suite
pytest

# With coverage report
pytest --cov=src --cov-report=html
```

## 3. Verify Training Runs

For model/trainer changes:
```bash
# Quick sanity check with CPU
python src/train.py trainer=cpu trainer.max_epochs=1
```

## 4. Git Commit

Follow Conventional Commits:
```bash
git add -A
git commit -m "type(scope): description"
# Types: feat, fix, docs, style, refactor, test, chore, perf, ci, build
# Example: "refactor(utils): modernize checkpoint loading with weights_only"
```

## 5. Documentation Updates

If changing:
- Architecture → Update `CLAUDE.md` and `README.md`
- Config structure → Update `configs/README.md` (if exists)
- Dependencies → Update `requirements.txt` and `pyproject.toml`

## Notes

- **No Claude attribution in commits**: Don't add Co-Authored-By lines
- **No force push**: Use `--force-with-lease` if needed
- **No sensitive data**: Never commit .env, secrets, logs, .claude/ directory