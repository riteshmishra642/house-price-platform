# Data Profiling

This report profiles `data/raw/ames_housing_raw.csv` as loaded by
`src/pipelines/data_ingestion.py`. Regenerate these numbers at any time with:

```bash
python -m src.pipelines.data_ingestion
```

(Exact values below reflect a run using the synthetic fallback dataset — see
"Why the dataset is suitable" below for how it maps to the real Ames Housing
dataset. Numbers will differ slightly once trained against a real
internet-downloaded copy.)

## 1. Shape & Data Types

| Metric | Value |
|---|---|
| Rows | 2,930 |
| Columns | 50 |
| Integer columns | 24 |
| Float columns | 4 |
| String/categorical columns | 22 |
| Duplicate rows | 0 |

## 2. Missing Values

| Column | Missing Count | Missing % | Why it's missing |
|---|---|---|---|
| Fence | 2,339 | 79.8% | Most properties have no fence (domain-specific: means "None", not unknown) |
| LotFrontage | 522 | 17.8% | Not recorded for some lots (e.g. cul-de-sac lots without a standard frontage) |
| GarageType | 181 | 6.2% | No garage present |
| GarageYrBlt | 181 | 6.2% | No garage present |
| GarageQual | 181 | 6.2% | No garage present |
| BsmtQual | 61 | 2.1% | No basement present |
| BsmtCond | 55 | 1.9% | No basement present |
| MasVnrArea | 24 | 0.8% | Masonry veneer not recorded/applicable |

**Data quality decision**: columns like `Fence`, `GarageType`, `BsmtQual` are
*not* imputed with a statistical mode — per the Ames data dictionary, a missing
value there means "this house doesn't have this feature." Imputing with the
mode (e.g. filling missing `BsmtQual` with `"TA"`) would fabricate a basement
that doesn't exist. See `NONE_MEANS_ABSENT_COLUMNS` in
`src/preprocessing/cleaning.py` for the full list and handling logic.

## 3. Target Variable (SalePrice) Distribution

| Statistic | Value |
|---|---|
| Mean | $236,681 |
| Median | $227,350 |
| Std. Dev. | $67,605 |
| Min | $69,300 |
| Max | $483,400 |
| Skewness | 0.61 (right-skewed) |

**Actionable insight**: the right skew (a small number of very expensive homes
pulling the mean above the median) is why the modeling pipeline trains on
`log1p(SalePrice)` instead of raw dollars (see `src/training/trainer.py`) —
this makes the loss function penalize proportional errors consistently across
cheap and expensive homes, rather than letting a handful of expensive outliers
dominate the loss.

## 4. Outlier Summary

Outliers are detected via the IQR method (`1.5x` for exploratory profiling,
`3.0x` — configurable in `config.yaml` — for the production cleaning pipeline,
which intentionally caps rather than removes them). A representative cleaning
run flagged and capped outliers in 5 columns:

| Column | Outliers Capped |
|---|---|
| MasVnrArea | 43 |
| LotArea | 17 |
| OpenPorchSF | 24 |
| WoodDeckSF | 22 |
| LotFrontage | 1 |

**Why cap instead of delete**: deleting outlier rows loses real information —
a $480,000 estate sale is a legitimate transaction, not a data error. Capping
(winsorizing) at the IQR boundary keeps the row but prevents a single extreme
value from disproportionately influencing linear model coefficients.

## 5. Correlation Summary (numeric features vs. SalePrice)

| Feature | Correlation with SalePrice |
|---|---|
| GrLivArea | 0.53 |
| 2ndFlrSF | 0.41 |
| 1stFlrSF | 0.34 |
| GarageArea | 0.25 |
| GarageCars | 0.25 |
| TotalBsmtSF | 0.24 |
| GarageYrBlt | 0.17 |
| OverallQual | 0.17 |
| YearBuilt | 0.16 |
| YearRemodAdd | 0.16 |

**Business insight**: living area and floor square footage dominate the
correlation ranking — consistent with how real estate is priced ("price per
square foot") — while garage and basement size are secondary but still
meaningful drivers. See `reports/figures/eda/04_correlation_heatmap_full.png`
and `05_correlation_heatmap_top.png` for the full visual breakdown, and
`src/feature_engineering/selection.py` for how this compares against
nonlinear methods (mutual information, tree/permutation importance).

## 6. Data Quality Issues Identified & Resolved

| Issue | Resolution | Module |
|---|---|---|
| Domain-specific missingness (no basement/garage/fence) | Filled with `"None"` / `0`, not mode/median | `src/preprocessing/cleaning.py` |
| Statistical missingness (LotFrontage, MasVnrArea) | Median (numeric) / mode (categorical) imputation, learned from training data only | `src/preprocessing/cleaning.py` |
| Extreme outliers (large lots, porches) | IQR-based capping (winsorization) | `src/preprocessing/cleaning.py` |
| Right-skewed target | log1p transform before training | `src/training/trainer.py` |
| Whitespace in string fields | Stripped | `src/preprocessing/cleaning.py` |
| Semantically-categorical numeric codes (MSSubClass) | Cast to string before encoding | `src/preprocessing/cleaning.py` |
| Unseen categories at inference time | `OneHotEncoder(handle_unknown="ignore")` | `src/preprocessing/pipeline.py` |

## 7. Why the Ames Housing Dataset Is Suitable

- **Size**: 2,930 rows is enough for meaningful train/test/CV splits without
  needing deep learning-scale data.
- **Feature richness**: ~79 explanatory variables spanning size, quality,
  location, age, and amenities — enough to demonstrate real feature
  engineering and selection decisions, not just a handful of columns.
- **Realistic messiness**: genuine missingness patterns with clear domain
  semantics (not-applicable vs. truly-unknown), which is exactly the kind of
  judgment call a production data cleaning pipeline must make.
- **Well-documented data dictionary**: every column's meaning is externally
  documented, which is what allows the domain-specific missingness handling
  in this project to be defensible rather than guessed.
