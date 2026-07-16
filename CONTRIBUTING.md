# Contributing

Thanks for considering a contribution to the House Price Prediction Platform.

## Development setup

```bash
git clone <repo-url>
cd house-price-platform
poetry install --with dev
# or: pip install -r requirements.txt
```

## Before opening a pull request

1. **Format your code:**
   ```bash
   black src api dashboard tests
   isort src api dashboard tests
   ```
2. **Lint:**
   ```bash
   flake8 src api dashboard tests
   ```
3. **Run the test suite:**
   ```bash
   python -m src.training.trainer   # train a model first if you haven't
   pytest --cov=src --cov=api
   ```
4. Keep commits focused and use descriptive commit messages (see git log for
   the convention used throughout this project: `feat:`, `fix:`, `docs:`,
   `chore:`, `test:`).

## Project conventions

- All configuration lives in `config/config.yaml` — avoid hardcoding paths,
  hyperparameters, or column names in module code.
- Every module that has a meaningful standalone behavior should have a
  runnable `if __name__ == "__main__":` block for quick manual testing.
- Optional dependencies (xgboost, lightgbm, catboost, shap, optuna, mlflow,
  plotly) must be imported defensively (`try/except ImportError`) so the rest
  of the platform keeps working without them — follow the existing pattern in
  `src/training/model_factory.py`.
- New features should come with corresponding tests in `tests/`.

## Reporting issues

Please include:
- What you expected to happen vs. what actually happened
- Steps to reproduce
- Python version and OS
- Relevant logs from `reports/logs/app.log`
