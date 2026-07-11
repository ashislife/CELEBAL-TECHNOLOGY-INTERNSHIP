import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Energy Forecast",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Energy Consumption Forecast")
st.caption(
    "XGBoost-based consumption forecasting using smart-meter, "
    "weather, household and historical usage features."
)


# =========================================================
# FILE PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "feature_dataset.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "energy_forecast_model.pkl"
)

FEATURES_PATH = os.path.join(
    BASE_DIR,
    "models",
    "features.pkl"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    trained_model = joblib.load(MODEL_PATH)
    trained_features = joblib.load(FEATURES_PATH)

    return trained_model, trained_features


# Dashboard ke liye limited rows load karenge.
# Full dataset model training ke liye already use ho chuka hai.
@st.cache_data
def load_forecast_data():
    required_columns = [
        "LCLid",
        "day",
        "energy_sum",
        "energy_median",
        "energy_mean",
        "energy_max",
        "energy_count",
        "energy_std",
        "energy_min",
        "stdorToU",
        "Acorn",
        "Acorn_grouped",
        "icon",
        "precipType",
        "summary",
        "season",
        "temperatureMax",
        "humidity",
        "windSpeed",
        "pressure",
        "year",
        "month",
        "day_of_month",
        "day_of_week",
        "week",
        "quarter",
        "is_weekend",
        "lag_1",
        "lag_7",
        "lag_30",
        "rolling_mean_7",
        "rolling_std_7"
    ]

    return pd.read_csv(
        DATA_PATH,
        usecols=required_columns,
        parse_dates=["day"],
        nrows=300000,
        low_memory=False
    )


model, feature_names = load_model()
df = load_forecast_data()


# =========================================================
# PREPARE FEATURES EXACTLY LIKE TRAINING
# =========================================================

@st.cache_data
def prepare_model_features(raw_df, trained_feature_names):

    model_df = raw_df.drop(
        columns=["energy_sum", "LCLid", "day"],
        errors="ignore"
    ).copy()

    categorical_columns = [
        "stdorToU",
        "Acorn",
        "Acorn_grouped",
        "icon",
        "precipType",
        "summary",
        "season"
    ]

    existing_categorical_columns = [
        column
        for column in categorical_columns
        if column in model_df.columns
    ]

    model_df = pd.get_dummies(
        model_df,
        columns=existing_categorical_columns,
        drop_first=True
    )

    # Training ke exact 150 columns aur exact order restore hoga.
    model_df = model_df.reindex(
        columns=trained_feature_names,
        fill_value=0
    )

    return model_df


with st.spinner("Preparing forecast data..."):
    X = prepare_model_features(df, feature_names)
    y_actual = df["energy_sum"]

    predictions = model.predict(X)


results = pd.DataFrame({
    "Household": df["LCLid"],
    "Date": df["day"],
    "Actual Consumption": y_actual,
    "Predicted Consumption": predictions
})

results["Absolute Error"] = (
    results["Actual Consumption"]
    - results["Predicted Consumption"]
).abs()


# =========================================================
# MODEL METRICS
# =========================================================

mae = mean_absolute_error(
    results["Actual Consumption"],
    results["Predicted Consumption"]
)

rmse = np.sqrt(
    mean_squared_error(
        results["Actual Consumption"],
        results["Predicted Consumption"]
    )
)

r2 = r2_score(
    results["Actual Consumption"],
    results["Predicted Consumption"]
)


metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric(
    "Records Evaluated",
    f"{len(results):,}"
)

metric2.metric(
    "MAE",
    f"{mae:.3f} kWh"
)

metric3.metric(
    "RMSE",
    f"{rmse:.3f} kWh"
)

metric4.metric(
    "R² Score",
    f"{r2:.4f}"
)


st.divider()


# =========================================================
# HOUSEHOLD FILTER
# =========================================================

st.subheader("Household Forecast Analysis")

households = sorted(
    results["Household"].dropna().unique().tolist()
)

selected_household = st.selectbox(
    "Select Household",
    households
)

household_result = (
    results[
        results["Household"] == selected_household
    ]
    .sort_values("Date")
    .copy()
)


if household_result.empty:
    st.warning("No forecast data found for this household.")

else:
    average_actual = household_result[
        "Actual Consumption"
    ].mean()

    average_prediction = household_result[
        "Predicted Consumption"
    ].mean()

    average_error = household_result[
        "Absolute Error"
    ].mean()

    h1, h2, h3 = st.columns(3)

    h1.metric(
        "Average Actual",
        f"{average_actual:.2f} kWh"
    )

    h2.metric(
        "Average Predicted",
        f"{average_prediction:.2f} kWh"
    )

    h3.metric(
        "Average Error",
        f"{average_error:.2f} kWh"
    )

    trend_data = household_result.melt(
        id_vars=["Date"],
        value_vars=[
            "Actual Consumption",
            "Predicted Consumption"
        ],
        var_name="Consumption Type",
        value_name="Energy Consumption"
    )

    forecast_chart = px.line(
        trend_data,
        x="Date",
        y="Energy Consumption",
        color="Consumption Type",
        title=f"Actual vs Predicted Consumption — {selected_household}"
    )

    forecast_chart.update_layout(
        xaxis_title="Date",
        yaxis_title="Energy Consumption (kWh)",
        legend_title=""
    )

    st.plotly_chart(
        forecast_chart,
        use_container_width=True
    )


# =========================================================
# OVERALL ACTUAL VS PREDICTED TREND
# =========================================================

st.divider()
st.subheader("Overall Forecast Trend")

daily_forecast = (
    results
    .groupby("Date")[
        [
            "Actual Consumption",
            "Predicted Consumption"
        ]
    ]
    .mean()
    .reset_index()
)

daily_forecast_long = daily_forecast.melt(
    id_vars=["Date"],
    value_vars=[
        "Actual Consumption",
        "Predicted Consumption"
    ],
    var_name="Consumption Type",
    value_name="Average Energy Consumption"
)

overall_chart = px.line(
    daily_forecast_long,
    x="Date",
    y="Average Energy Consumption",
    color="Consumption Type",
    title="Average Actual vs Predicted Energy Consumption"
)

overall_chart.update_layout(
    xaxis_title="Date",
    yaxis_title="Average Consumption (kWh)",
    legend_title=""
)

st.plotly_chart(
    overall_chart,
    use_container_width=True
)


# =========================================================
# PREDICTION ACCURACY SCATTER PLOT
# =========================================================

st.divider()
st.subheader("Prediction Accuracy")

scatter_sample = results.sample(
    n=min(5000, len(results)),
    random_state=42
)

accuracy_chart = px.scatter(
    scatter_sample,
    x="Actual Consumption",
    y="Predicted Consumption",
    opacity=0.45,
    title="Actual Consumption vs Predicted Consumption",
    hover_data=["Household", "Date", "Absolute Error"]
)

accuracy_chart.update_layout(
    xaxis_title="Actual Consumption (kWh)",
    yaxis_title="Predicted Consumption (kWh)"
)

st.plotly_chart(
    accuracy_chart,
    use_container_width=True
)


# =========================================================
# FORECAST TABLE AND DOWNLOAD
# =========================================================

st.divider()
st.subheader("Forecast Records")

display_columns = [
    "Household",
    "Date",
    "Actual Consumption",
    "Predicted Consumption",
    "Absolute Error"
]

st.dataframe(
    household_result[display_columns].round(3),
    use_container_width=True,
    hide_index=True
)

csv_data = household_result[
    display_columns
].to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Household Forecast",
    data=csv_data,
    file_name=f"{selected_household}_energy_forecast.csv",
    mime="text/csv"
)