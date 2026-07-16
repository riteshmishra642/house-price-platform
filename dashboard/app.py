"""
Streamlit dashboard: the non-technical-user-facing interface for the House
Price Prediction Platform.

Run locally with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Ensure the project root is on sys.path when Streamlit runs this file
# directly (streamlit run dashboard/app.py), not as a package import.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prediction.predictor import get_predictor  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402

config = load_config()

st.set_page_config(
    page_title=config.dashboard.title,
    page_icon=config.dashboard.page_icon,
    layout=config.dashboard.layout,
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .prediction-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-card h1 {
        font-size: 2.75rem;
        margin: 0;
        color: white;
    }
    .prediction-card p {
        opacity: 0.85;
        margin-top: 0.25rem;
    }
    .metric-box {
        background-color: rgba(37, 99, 235, 0.08);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading trained model...")
def load_predictor():
    return get_predictor()


@st.cache_data(show_spinner=False)
def load_metrics():
    import json

    with open(resolve_path(config.paths.metrics_file), "r", encoding="utf-8") as f:
        return json.load(f)


NEIGHBORHOODS = [
    "NAmes",
    "CollgCr",
    "OldTown",
    "Edwards",
    "Somerst",
    "Gilbert",
    "NridgHt",
    "Sawyer",
    "NWAmes",
    "SawyerW",
    "BrkSide",
    "Crawfor",
    "Mitchel",
    "NoRidge",
    "Timber",
    "IDOTRR",
    "ClearCr",
    "StoneBr",
    "SWISU",
    "Blmngtn",
]
QUALITY_OPTIONS = ["Po", "Fa", "TA", "Gd", "Ex"]
HOUSE_STYLES = ["1Story", "2Story", "1.5Fin", "SLvl", "SFoyer"]
BLDG_TYPES = ["1Fam", "2fmCon", "Duplex", "TwnhsE", "Twnhs"]


def sidebar_property_form() -> dict:
    """Beautiful sidebar with a full property input form."""
    st.sidebar.title(f"{config.dashboard.page_icon} Property Details")
    st.sidebar.caption("Enter the property's attributes, then scroll down for the prediction.")

    with st.sidebar.expander("📍 Location & Type", expanded=True):
        neighborhood = st.selectbox("Neighborhood", NEIGHBORHOODS, index=1)
        bldg_type = st.selectbox("Building Type", BLDG_TYPES, index=0)
        house_style = st.selectbox("House Style", HOUSE_STYLES, index=1)
        ms_zoning = st.selectbox("Zoning", ["RL", "RM", "FV", "RH", "C (all)"], index=0)

    with st.sidebar.expander("📐 Size & Layout", expanded=True):
        lot_area = st.number_input("Lot Area (sq ft)", min_value=500, max_value=250000, value=9600, step=100)
        lot_frontage = st.number_input("Lot Frontage (ft)", min_value=0.0, max_value=500.0, value=80.0, step=1.0)
        gr_liv_area = st.number_input(
            "Above-Grade Living Area (sq ft)", min_value=200, max_value=6000, value=1710, step=10
        )
        total_bsmt_sf = st.number_input("Total Basement Area (sq ft)", min_value=0, max_value=6000, value=856, step=10)
        first_flr_sf = st.number_input("1st Floor Area (sq ft)", min_value=0, max_value=4000, value=856, step=10)
        second_flr_sf = st.number_input("2nd Floor Area (sq ft)", min_value=0, max_value=3000, value=854, step=10)

    with st.sidebar.expander("🛏️ Rooms", expanded=False):
        bedrooms = st.slider("Bedrooms", 0, 10, 3)
        full_bath = st.slider("Full Bathrooms", 0, 5, 2)
        half_bath = st.slider("Half Bathrooms", 0, 3, 1)
        bsmt_full_bath = st.slider("Basement Full Bathrooms", 0, 3, 1)
        tot_rooms = st.slider("Total Rooms Above Grade", 2, 16, 8)
        fireplaces = st.slider("Fireplaces", 0, 4, 1)

    with st.sidebar.expander("⭐ Quality & Condition", expanded=False):
        overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 7)
        overall_cond = st.slider("Overall Condition (1-10)", 1, 10, 5)
        exter_qual = st.select_slider("Exterior Quality", QUALITY_OPTIONS, value="Gd")
        kitchen_qual = st.select_slider("Kitchen Quality", QUALITY_OPTIONS, value="Gd")
        bsmt_qual = st.select_slider("Basement Quality", QUALITY_OPTIONS, value="Gd")
        heating_qc = st.select_slider("Heating Quality", QUALITY_OPTIONS, value="Ex")

    with st.sidebar.expander("🚗 Garage & Extras", expanded=False):
        garage_cars = st.slider("Garage Capacity (cars)", 0, 5, 2)
        garage_area = st.number_input("Garage Area (sq ft)", min_value=0, max_value=1500, value=548, step=10)
        wood_deck_sf = st.number_input("Wood Deck Area (sq ft)", min_value=0, max_value=1000, value=0, step=10)
        open_porch_sf = st.number_input("Open Porch Area (sq ft)", min_value=0, max_value=600, value=61, step=5)
        pool_area = st.number_input("Pool Area (sq ft)", min_value=0, max_value=1000, value=0, step=10)

    with st.sidebar.expander("📅 Dates", expanded=False):
        year_built = st.slider("Year Built", 1872, 2026, 2003)
        year_remod = st.slider("Year Remodeled", year_built, 2026, max(year_built, 2003))
        mo_sold = st.slider("Month Sold", 1, 12, 6)
        yr_sold = st.slider("Year Sold", 2000, 2026, 2008)

    return {
        "MSSubClass": "60",
        "MSZoning": ms_zoning,
        "LotFrontage": lot_frontage,
        "LotArea": lot_area,
        "Street": "Pave",
        "LotShape": "Reg",
        "LandContour": "Lvl",
        "Utilities": "AllPub",
        "Neighborhood": neighborhood,
        "BldgType": bldg_type,
        "HouseStyle": house_style,
        "OverallQual": overall_qual,
        "OverallCond": overall_cond,
        "YearBuilt": year_built,
        "YearRemodAdd": year_remod,
        "RoofStyle": "Gable",
        "Exterior1st": "VinylSd",
        "MasVnrArea": 0.0,
        "ExterQual": exter_qual,
        "Foundation": "PConc",
        "BsmtQual": bsmt_qual,
        "BsmtCond": "TA",
        "TotalBsmtSF": total_bsmt_sf,
        "HeatingQC": heating_qc,
        "CentralAir": "Y",
        "1stFlrSF": first_flr_sf,
        "2ndFlrSF": second_flr_sf,
        "GrLivArea": gr_liv_area,
        "BsmtFullBath": bsmt_full_bath,
        "FullBath": full_bath,
        "HalfBath": half_bath,
        "BedroomAbvGr": bedrooms,
        "KitchenQual": kitchen_qual,
        "TotRmsAbvGrd": tot_rooms,
        "Fireplaces": fireplaces,
        "GarageType": "Attchd" if garage_cars > 0 else "None",
        "GarageYrBlt": float(year_built) if garage_cars > 0 else None,
        "GarageCars": garage_cars,
        "GarageArea": garage_area,
        "GarageQual": "TA" if garage_cars > 0 else "None",
        "WoodDeckSF": wood_deck_sf,
        "OpenPorchSF": open_porch_sf,
        "PoolArea": pool_area,
        "Fence": "None",
        "MoSold": mo_sold,
        "YrSold": yr_sold,
        "SaleType": "WD",
        "SaleCondition": "Normal",
    }


def render_price_gauge(predicted_price: float, low: float, high: float) -> None:
    """Simple, dependency-light gauge built from Streamlit's own progress bar
    (avoids requiring plotly just for the dashboard, since plotly is treated
    as an optional dependency elsewhere in the platform)."""
    st.markdown("**Price Range Context**")
    span = max(high - low, 1.0)
    position = (predicted_price - low) / span
    st.progress(min(max(position, 0.0), 1.0))
    col1, col2, col3 = st.columns(3)
    col1.markdown(f"<div class='metric-box'>Low<br><b>${low:,.0f}</b></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-box'>Estimate<br><b>${predicted_price:,.0f}</b></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='metric-box'>High<br><b>${high:,.0f}</b></div>", unsafe_allow_html=True)


def render_prediction_tab(property_input: dict) -> None:
    st.subheader("💰 Price Prediction")

    try:
        predictor = load_predictor()
    except Exception as exc:  # noqa: BLE001
        st.error("No trained model found. Train one first by running:\n\n" "`python -m src.training.trainer`")
        st.exception(exc)
        return

    explain_toggle = st.checkbox("Include explanation (requires SHAP)", value=False)

    if st.button("🔮 Predict Price", type="primary", use_container_width=True):
        with st.spinner("Computing prediction..."):
            result = predictor.predict(property_input, explain=explain_toggle)

        st.markdown(
            f"""
            <div class="prediction-card">
                <p>Estimated Sale Price</p>
                <h1>${result.predicted_price:,.0f}</h1>
                <p>Model: {result.model_name} | Confidence range shown below</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_price_gauge(result.predicted_price, result.confidence_interval_low, result.confidence_interval_high)

        if explain_toggle and result.explanation:
            st.markdown("**Why this price? (SHAP feature contributions)**")
            if "error" in result.explanation:
                st.info(result.explanation["error"])
            else:
                contributions = result.explanation["feature_contributions"]
                top_items = list(contributions.items())[:12]
                contrib_df = pd.DataFrame(top_items, columns=["Feature", "SHAP Contribution"])
                st.bar_chart(contrib_df.set_index("Feature"))


def render_market_analysis_tab() -> None:
    st.subheader("📊 Market Analysis")
    try:
        from src.feature_engineering.features import engineer_features
        from src.pipelines.data_ingestion import load_or_create_raw_dataset
        from src.preprocessing.cleaning import clean_dataset

        raw_df = load_or_create_raw_dataset()
        cleaned_df, _ = clean_dataset(raw_df)
        df = engineer_features(cleaned_df)
    except Exception as exc:  # noqa: BLE001
        st.warning("Could not load dataset for market analysis.")
        st.exception(exc)
        return

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Median Sale Price by Neighborhood**")
        by_neighborhood = df.groupby("Neighborhood")["SalePrice"].median().sort_values(ascending=False)
        st.bar_chart(by_neighborhood)
    with col2:
        st.markdown("**Sale Price vs. Living Area**")
        st.scatter_chart(df[["GrLivArea", "SalePrice"]], x="GrLivArea", y="SalePrice")

    st.markdown("**Sale Price Distribution**")
    st.bar_chart(np.histogram(df["SalePrice"], bins=30)[0])


def render_model_comparison_tab() -> None:
    st.subheader("🧠 Model Comparison")
    try:
        metrics = load_metrics()
    except FileNotFoundError:
        st.warning("No metrics found yet. Train models first with `python -m src.training.trainer`.")
        return

    metrics_df = pd.DataFrame(metrics).sort_values("rmse")
    st.dataframe(
        metrics_df[["name", "rmse", "mae", "r2", "mape", "cv_rmse_mean"]].rename(
            columns={
                "name": "Model",
                "rmse": "RMSE ($)",
                "mae": "MAE ($)",
                "r2": "R²",
                "mape": "MAPE (%)",
                "cv_rmse_mean": "CV RMSE",
            }
        ),
        use_container_width=True,
    )
    st.markdown("**Test RMSE by Model (lower is better)**")
    st.bar_chart(metrics_df.set_index("name")["rmse"])


def main() -> None:
    st.title(f"{config.dashboard.page_icon} {config.dashboard.title}")
    st.caption(
        "Enterprise-grade house price prediction with explainable AI. "
        "Enter property details in the sidebar, then predict."
    )

    property_input = sidebar_property_form()

    tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Market Analysis", "🧠 Model Comparison"])
    with tab1:
        render_prediction_tab(property_input)
    with tab2:
        render_market_analysis_tab()
    with tab3:
        render_model_comparison_tab()


if __name__ == "__main__":
    main()
