"""
Feature selection module.

Compares multiple feature-importance/selection methods side by side rather
than trusting a single method blindly -- correlation, mutual information,
tree-based importance, permutation importance, and RFE each have different
blind spots (e.g. correlation misses nonlinear relationships; tree
importance is biased toward high-cardinality features), so cross-checking
them produces a more defensible final feature set.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import (
    RFE,
    VarianceThreshold,
    mutual_info_regression,
)
from sklearn.inspection import permutation_importance

from src.preprocessing.pipeline import build_full_preprocessor
from src.utils.logger import get_logger

logger = get_logger(__name__)


def correlation_ranking(df: pd.DataFrame, target: str = "SalePrice") -> pd.Series:
    """Simple, fast, but linear-only: misses nonlinear/threshold effects
    (e.g. a quality jump from 'TA' to 'Gd' is not linear in the encoding)."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()[target].drop(target, errors="ignore").abs()
    return corr.sort_values(ascending=False)


def mutual_information_ranking(X: pd.DataFrame, y: pd.Series, random_seed: int = 42) -> pd.Series:
    """
    Mutual information captures nonlinear dependence that Pearson correlation
    misses (e.g. Neighborhood's effect on price is categorical/nonlinear).
    Only applied to numeric-encoded columns here; categorical columns should
    be label/frequency-encoded before calling this in a full pipeline.
    """
    numeric_X = X.select_dtypes(include=[np.number]).fillna(0)
    mi_scores = mutual_info_regression(numeric_X, y, random_state=random_seed)
    return pd.Series(mi_scores, index=numeric_X.columns).sort_values(ascending=False)


def tree_based_importance(X: pd.DataFrame, y: pd.Series, random_seed: int = 42) -> pd.Series:
    """
    Random Forest impurity-based importance: fast and captures nonlinear
    interactions, but biased toward high-cardinality / high-variance
    features -- cross-check against permutation importance below.
    """
    preprocessor = build_full_preprocessor(pd.concat([X, y.rename("SalePrice")], axis=1))
    X_transformed = preprocessor.fit_transform(X)
    model = RandomForestRegressor(n_estimators=200, max_depth=12, n_jobs=-1, random_state=random_seed)
    model.fit(X_transformed, np.log1p(y))
    importances = pd.Series(model.feature_importances_, index=X_transformed.columns)
    return importances.sort_values(ascending=False)


def permutation_importance_ranking(
    X: pd.DataFrame, y: pd.Series, random_seed: int = 42, n_repeats: int = 5
) -> pd.Series:
    """
    Measures the actual drop in model performance when a feature's values
    are shuffled -- unlike impurity-based importance, this is unbiased with
    respect to feature cardinality and directly reflects predictive value.
    More expensive to compute, which is why it's run after cheaper filters
    have already narrowed the candidate set in a real pipeline.
    """
    preprocessor = build_full_preprocessor(pd.concat([X, y.rename("SalePrice")], axis=1))
    X_transformed = preprocessor.fit_transform(X)
    model = RandomForestRegressor(n_estimators=150, max_depth=10, n_jobs=-1, random_state=random_seed)
    model.fit(X_transformed, np.log1p(y))

    result = permutation_importance(
        model, X_transformed, np.log1p(y), n_repeats=n_repeats, random_state=random_seed, n_jobs=-1
    )
    return pd.Series(result.importances_mean, index=X_transformed.columns).sort_values(ascending=False)


def recursive_feature_elimination(
    X: pd.DataFrame, y: pd.Series, n_features_to_select: int = 20, random_seed: int = 42
) -> List[str]:
    """
    RFE repeatedly fits a model and drops the weakest feature(s), directly
    optimizing for a target feature-count budget -- most useful when there's
    a hard constraint (e.g. a downstream system that can only accept N
    inputs), rather than as a primary exploratory tool.
    """
    preprocessor = build_full_preprocessor(pd.concat([X, y.rename("SalePrice")], axis=1))
    X_transformed = preprocessor.fit_transform(X)
    model = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=random_seed)

    selector = RFE(model, n_features_to_select=n_features_to_select, step=0.1)
    selector.fit(X_transformed, np.log1p(y))
    selected = X_transformed.columns[selector.support_].tolist()
    return selected


def variance_threshold_filter(X: pd.DataFrame, threshold: float = 0.0) -> List[str]:
    """Drops near-constant numeric features that carry almost no signal
    (e.g. Utilities, which is 'AllPub' for ~99.9% of Ames properties)."""
    numeric_X = X.select_dtypes(include=[np.number]).fillna(0)
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(numeric_X)
    return numeric_X.columns[selector.get_support()].tolist()


def compare_feature_selection_methods(
    X: pd.DataFrame, y: pd.Series, top_k: int = 15, random_seed: int = 42
) -> pd.DataFrame:
    """
    Runs every method and returns a single comparison table: for each
    method's top-K features, which features appear in multiple methods'
    top lists (a strong "keep this feature" signal) vs. only one (worth a
    second look before discarding or keeping).
    """
    logger.info("Running feature selection comparison across 4 methods...")

    corr_top = set(correlation_ranking(pd.concat([X, y.rename("SalePrice")], axis=1)).head(top_k).index)
    mi_top = set(mutual_information_ranking(X, y, random_seed).head(top_k).index)
    tree_top = set(tree_based_importance(X, y, random_seed).head(top_k).index)
    perm_top = set(permutation_importance_ranking(X, y, random_seed, n_repeats=3).head(top_k).index)

    all_features = corr_top | mi_top | tree_top | perm_top
    comparison = pd.DataFrame(
        {
            "correlation": [f in corr_top for f in all_features],
            "mutual_information": [f in mi_top for f in all_features],
            "tree_importance": [f in tree_top for f in all_features],
            "permutation_importance": [f in perm_top for f in all_features],
        },
        index=list(all_features),
    )
    comparison["agreement_count"] = comparison.sum(axis=1)
    comparison = comparison.sort_values("agreement_count", ascending=False)
    return comparison


if __name__ == "__main__":
    from src.feature_engineering.features import engineer_features
    from src.pipelines.data_ingestion import load_or_create_raw_dataset
    from src.preprocessing.cleaning import clean_dataset

    raw_df = load_or_create_raw_dataset()
    cleaned_df, _ = clean_dataset(raw_df)
    engineered_df = engineer_features(cleaned_df)

    X = engineered_df.drop(columns=["SalePrice", "Id"])
    y = engineered_df["SalePrice"]

    comparison_table = compare_feature_selection_methods(X, y, top_k=15)
    print(comparison_table)
