# Changelog

All notable changes to this project are documented in this file.

## [1.0.0] - Initial Release

### Added
- Business understanding documentation (`docs/01_business_understanding.md`)
- Data ingestion pipeline with OpenML/URL download and synthetic fallback generator
- Data cleaning module with domain-aware missingness handling and outlier capping
- Feature engineering module (19+ derived features: age, ratios, quality indices, luxury score)
- Feature selection comparison (correlation, mutual information, tree importance, permutation importance, RFE)
- Model training pipeline comparing 14 regression models (linear, tree-based, boosting, ensembles)
- Hyperparameter tuning via RandomizedSearchCV and Optuna
- Full evaluation diagnostics (residuals, prediction error, learning curves, validation curves)
- 80+ figure EDA visualization suite (static + interactive Plotly dashboards)
- SHAP-based explainability module (summary, waterfall, dependence, force, decision plots)
- Production FastAPI service (`/predict`, `/batch_predict`, `/health`, `/model_info`, `/metrics`)
- Streamlit dashboard with property input form, prediction card, market analysis, and model comparison tabs
- MLflow experiment tracking integration
- Full pytest test suite (cleaning, feature engineering, preprocessing, model factory, prediction, API)
- Docker + docker-compose multi-stage build (trainer, API, dashboard)
- GitHub Actions CI/CD pipeline (lint, test with coverage, Docker build)

### Design notes
- All optional dependencies (xgboost, lightgbm, catboost, shap, optuna, mlflow, plotly) degrade
  gracefully when not installed, so the core platform always runs.
- Synthetic data fallback ensures full reproducibility in offline/restricted environments.
