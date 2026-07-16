"""
Exploratory Data Analysis visualization module.

Generates the full EDA figure set (target distribution, correlations,
categorical comparisons, per-feature relationships) as static PNGs (via
matplotlib/seaborn) plus a couple of interactive Plotly HTML dashboards.

Design choice: one function per *category* of chart (not one per single
column) that loops over relevant columns internally. This keeps the module
maintainable while still producing 50+ individual chart files, matching how
a real EDA notebook is organized (grouped by analysis theme, not one giant
flat script).
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import plotly.express as px
    import plotly.graph_objects as go

    _HAS_PLOTLY = True
except ImportError:  # pragma: no cover
    _HAS_PLOTLY = False
    logger.warning("plotly not installed - interactive HTML dashboards will be skipped.")

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.autolayout": True, "font.size": 9})

KEY_NUMERIC_COLS = [
    "SalePrice", "GrLivArea", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF",
    "LotArea", "LotFrontage", "GarageArea", "OverallQual", "OverallCond",
    "YearBuilt", "YearRemodAdd", "TotalBathrooms", "TotRmsAbvGrd",
    "Fireplaces", "GarageCars", "WoodDeckSF", "OpenPorchSF",
]
KEY_CATEGORICAL_COLS = [
    "Neighborhood", "HouseStyle", "BldgType", "MSZoning", "SaleCondition",
    "Foundation", "ExterQual", "KitchenQual", "CentralAir", "GarageType",
]


def _figures_dir() -> Path:
    config = load_config()
    figures_dir = resolve_path(config.paths.figures_dir) / "eda"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def _save(fig, name: str) -> Path:
    path = _figures_dir() / name
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_target_distribution(df: pd.DataFrame, target: str = "SalePrice") -> List[Path]:
    """
    Business insight: shows the typical price range and how many homes sell
    far above/below the median (useful for market-segment framing).
    Technical insight: SalePrice is right-skewed, which is exactly why we
    train on log1p(SalePrice) in the modeling stage.
    Actionable insight: log-transform confirmed necessary before modeling.
    """
    paths = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(df[target], kde=True, ax=axes[0], color="#2563eb")
    axes[0].set_title(f"{target} Distribution (raw)")
    sns.histplot(np.log1p(df[target]), kde=True, ax=axes[1], color="#16a34a")
    axes[1].set_title(f"log1p({target}) Distribution")
    paths.append(_save(fig, "01_target_distribution.png"))

    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    sns.boxplot(x=df[target], ax=ax2, color="#f59e0b")
    ax2.set_title(f"{target} Boxplot (outlier check)")
    paths.append(_save(fig2, "02_target_boxplot.png"))
    return paths


def plot_missing_value_heatmap(df: pd.DataFrame) -> Path:
    """Visualizes where missingness concentrates -- useful before deciding
    domain-specific vs. statistical imputation strategy (see cleaning.py)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(df.isna(), cbar=False, cmap="rocket", ax=ax)
    ax.set_title("Missing Value Heatmap (rows x columns)")
    return _save(fig, "03_missing_value_heatmap.png")


def plot_correlation_heatmap(df: pd.DataFrame, target: str = "SalePrice") -> List[Path]:
    """Full correlation matrix + a focused top-N-correlated-with-target view,
    the second being what actually drives feature-selection decisions."""
    paths = []
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, cbar_kws={"shrink": 0.6})
    ax.set_title("Full Correlation Heatmap")
    paths.append(_save(fig, "04_correlation_heatmap_full.png"))

    top_corr = corr[target].abs().sort_values(ascending=False).head(16).index
    fig2, ax2 = plt.subplots(figsize=(9, 7))
    sns.heatmap(numeric_df[top_corr].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax2)
    ax2.set_title(f"Top Features Correlated with {target}")
    paths.append(_save(fig2, "05_correlation_heatmap_top.png"))
    return paths


def plot_numeric_histograms(df: pd.DataFrame, cols: List[str]) -> List[Path]:
    """One distribution histogram per key numeric feature."""
    paths = []
    for i, col in enumerate(cols, start=1):
        if col not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df[col], kde=True, ax=ax, color="#2563eb")
        ax.set_title(f"Distribution of {col}")
        paths.append(_save(fig, f"06_hist_{i:02d}_{col}.png"))
    return paths


def plot_numeric_boxplots(df: pd.DataFrame, cols: List[str]) -> List[Path]:
    """Boxplots surface outliers per feature, informing the IQR capping strategy."""
    paths = []
    for i, col in enumerate(cols, start=1):
        if col not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x=df[col], ax=ax, color="#f59e0b")
        ax.set_title(f"Boxplot of {col}")
        paths.append(_save(fig, f"07_box_{i:02d}_{col}.png"))
    return paths


def plot_scatter_vs_target(df: pd.DataFrame, cols: List[str], target: str = "SalePrice") -> List[Path]:
    """Scatter of each key numeric feature against SalePrice -- the most
    direct visual evidence of a feature's individual predictive power."""
    paths = []
    for i, col in enumerate(cols, start=1):
        if col not in df.columns or col == target:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x=df[col], y=df[target], alpha=0.4, ax=ax, color="#16a34a")
        ax.set_title(f"{col} vs {target}")
        paths.append(_save(fig, f"08_scatter_{i:02d}_{col}.png"))
    return paths


def plot_categorical_comparisons(df: pd.DataFrame, cols: List[str], target: str = "SalePrice") -> List[Path]:
    """
    Business insight per chart: which categories command a price premium.
    e.g. Neighborhood boxplots directly answer "does location matter, and
    by how much" -- one of the most common stakeholder questions.
    """
    paths = []
    for i, col in enumerate(cols, start=1):
        if col not in df.columns:
            continue
        order = df.groupby(col)[target].median().sort_values(ascending=False).index
        fig, ax = plt.subplots(figsize=(max(7, 0.5 * df[col].nunique()), 4.5))
        sns.boxplot(x=col, y=target, hue=col, data=df, order=order, ax=ax, palette="viridis", legend=False)
        ax.set_title(f"{target} by {col}")
        ax.tick_params(axis="x", rotation=45)
        paths.append(_save(fig, f"09_cat_box_{i:02d}_{col}.png"))
    return paths


def plot_countplots(df: pd.DataFrame, cols: List[str]) -> List[Path]:
    """Category frequency -- flags rare categories worth grouping (see
    rare_category_threshold in config.yaml)."""
    paths = []
    for i, col in enumerate(cols, start=1):
        if col not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(max(7, 0.4 * df[col].nunique()), 4))
        order = df[col].value_counts().index
        sns.countplot(x=col, hue=col, data=df, order=order, ax=ax, palette="crest", legend=False)
        ax.set_title(f"Count of Properties by {col}")
        ax.tick_params(axis="x", rotation=45)
        paths.append(_save(fig, f"10_count_{i:02d}_{col}.png"))
    return paths


def plot_violin_comparisons(df: pd.DataFrame, target: str = "SalePrice") -> List[Path]:
    """Violin plots reveal price-distribution shape differences that a
    boxplot's five-number summary can hide (e.g. bimodal subgroups)."""
    paths = []
    for i, col in enumerate(["OverallQual", "GarageCars", "Fireplaces", "BldgType"], start=1):
        if col not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.violinplot(x=col, y=target, hue=col, data=df, ax=ax, palette="mako", legend=False)
        ax.set_title(f"{target} Distribution by {col} (Violin)")
        paths.append(_save(fig, f"11_violin_{i:02d}_{col}.png"))
    return paths


def plot_pairplot(df: pd.DataFrame, cols: List[str], target: str = "SalePrice") -> Path:
    """Pairwise relationships among the strongest price drivers, to spot
    interaction effects (e.g. GrLivArea x OverallQual jointly)."""
    subset_cols = [c for c in cols if c in df.columns][:5] + [target]
    grid = sns.pairplot(df[subset_cols].sample(min(500, len(df)), random_state=42), corner=True)
    grid.fig.suptitle("Pairwise Relationships (top price drivers)", y=1.02)
    path = _figures_dir() / "12_pairplot_top_drivers.png"
    grid.savefig(path, dpi=110)
    plt.close(grid.fig)
    return path


def plot_year_built_analysis(df: pd.DataFrame, target: str = "SalePrice") -> List[Path]:
    """Construction-year trends: are newer homes worth more, and has the
    market's premium for new construction changed over time."""
    paths = []
    fig, ax = plt.subplots(figsize=(9, 4.5))
    yearly_median = df.groupby("YearBuilt")[target].median()
    ax.plot(yearly_median.index, yearly_median.values, color="#2563eb")
    ax.set_title(f"Median {target} by Year Built")
    ax.set_xlabel("Year Built")
    ax.set_ylabel(f"Median {target}")
    paths.append(_save(fig, "13_year_built_trend.png"))

    fig2, ax2 = plt.subplots(figsize=(9, 4.5))
    sns.scatterplot(x="HouseAge" if "HouseAge" in df.columns else "YearBuilt", y=target, data=df, alpha=0.4, ax=ax2)
    ax2.set_title(f"House Age vs {target}")
    paths.append(_save(fig2, "14_house_age_scatter.png"))
    return paths


def plot_garage_basement_analysis(df: pd.DataFrame, target: str = "SalePrice") -> List[Path]:
    """Garage/basement presence and size are consistently strong price
    drivers in real estate; these charts quantify that for this dataset."""
    paths = []
    if "GarageCars" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.boxplot(x="GarageCars", y=target, hue="GarageCars", data=df, ax=ax, palette="flare", legend=False)
        ax.set_title(f"{target} by Garage Capacity (cars)")
        paths.append(_save(fig, "15_garage_cars_box.png"))
    if "TotalBsmtSF" in df.columns:
        fig2, ax2 = plt.subplots(figsize=(7, 4.5))
        sns.scatterplot(x="TotalBsmtSF", y=target, data=df, alpha=0.4, ax=ax2, color="#7c3aed")
        ax2.set_title(f"Basement Size vs {target}")
        paths.append(_save(fig2, "16_basement_size_scatter.png"))
    return paths


def plot_interactive_neighborhood_dashboard(df: pd.DataFrame, target: str = "SalePrice") -> Path | None:
    """Interactive Plotly dashboard: hoverable neighborhood price comparison,
    useful for a live dashboard/report rather than a static image."""
    if not _HAS_PLOTLY:
        logger.warning("Skipping interactive neighborhood dashboard - plotly not installed.")
        return None
    fig = px.box(
        df, x="Neighborhood", y=target, color="Neighborhood",
        title=f"Interactive: {target} Distribution by Neighborhood",
        points=False,
    )
    fig.update_layout(showlegend=False, xaxis_tickangle=-45)
    path = _figures_dir() / "17_interactive_neighborhood_dashboard.html"
    fig.write_html(str(path))
    return path


def plot_interactive_scatter_matrix(df: pd.DataFrame, target: str = "SalePrice") -> Path | None:
    """Interactive scatter of size/quality/age against price, colored by
    quality, for exploratory drill-down in a browser."""
    if not _HAS_PLOTLY:
        logger.warning("Skipping interactive scatter matrix - plotly not installed.")
        return None
    sample = df.sample(min(800, len(df)), random_state=42)
    fig = px.scatter(
        sample, x="GrLivArea", y=target, color="OverallQual",
        size="TotalBsmtSF" if "TotalBsmtSF" in sample.columns else None,
        hover_data=["Neighborhood", "YearBuilt"],
        title="Interactive: Living Area vs Sale Price (colored by Overall Quality)",
    )
    path = _figures_dir() / "18_interactive_scatter_price_area_quality.html"
    fig.write_html(str(path))
    return path


def generate_full_eda_report(df: pd.DataFrame, target: str = "SalePrice") -> List[Path]:
    """Run every EDA chart category and return all saved figure paths (50+ files)."""
    all_paths: List[Path] = []
    all_paths += plot_target_distribution(df, target)
    all_paths.append(plot_missing_value_heatmap(df))
    all_paths += plot_correlation_heatmap(df, target)
    all_paths += plot_numeric_histograms(df, KEY_NUMERIC_COLS)
    all_paths += plot_numeric_boxplots(df, KEY_NUMERIC_COLS)
    all_paths += plot_scatter_vs_target(df, KEY_NUMERIC_COLS, target)
    all_paths += plot_categorical_comparisons(df, KEY_CATEGORICAL_COLS, target)
    all_paths += plot_countplots(df, KEY_CATEGORICAL_COLS)
    all_paths += plot_violin_comparisons(df, target)
    all_paths.append(plot_pairplot(df, ["GrLivArea", "OverallQual", "TotalBsmtSF", "GarageArea", "YearBuilt"], target))
    all_paths += plot_year_built_analysis(df, target)
    all_paths += plot_garage_basement_analysis(df, target)
    all_paths.append(plot_interactive_neighborhood_dashboard(df, target))
    all_paths.append(plot_interactive_scatter_matrix(df, target))

    all_paths = [p for p in all_paths if p is not None]
    logger.info("EDA report complete: %d figures generated in %s", len(all_paths), _figures_dir())
    return all_paths


if __name__ == "__main__":
    from src.feature_engineering.features import engineer_features
    from src.pipelines.data_ingestion import load_or_create_raw_dataset
    from src.preprocessing.cleaning import clean_dataset

    raw_df = load_or_create_raw_dataset()
    cleaned_df, _ = clean_dataset(raw_df)
    engineered_df = engineer_features(cleaned_df)

    paths = generate_full_eda_report(engineered_df)
    print(f"Generated {len(paths)} EDA figures.")
