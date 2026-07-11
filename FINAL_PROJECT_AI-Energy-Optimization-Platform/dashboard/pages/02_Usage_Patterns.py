import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(
    page_title="Usage Pattern Analysis",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Energy Usage Pattern Analysis")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "feature_dataset.csv")

@st.cache_data
def load_data():
    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["day"]
    )
    return df

df = load_data()

# -----------------------------
# Monthly Trend
# -----------------------------

st.subheader("Monthly Energy Consumption")

monthly = (
    df.groupby(["year","month"])["energy_sum"]
      .mean()
      .reset_index()
)

monthly["Date"] = pd.to_datetime(
    monthly["year"].astype(str)
    + "-"
    + monthly["month"].astype(str)
)

fig1 = px.line(
    monthly,
    x="Date",
    y="energy_sum",
    markers=True,
    title="Average Monthly Energy Consumption"
)

st.plotly_chart(fig1, use_container_width=True)

# -----------------------------
# Weekday vs Weekend
# -----------------------------

st.subheader("Weekday vs Weekend")

week = (
    df.groupby("is_weekend")["energy_sum"]
      .mean()
      .reset_index()
)

week["Type"] = week["is_weekend"].map(
    {
        0:"Weekday",
        1:"Weekend"
    }
)

fig2 = px.bar(
    week,
    x="Type",
    y="energy_sum",
    text_auto=".2f",
    title="Average Energy Consumption"
)

st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# ACORN Groups
# -----------------------------

st.subheader("ACORN Group Analysis")

acorn = (
    df.groupby("Acorn_grouped")["energy_sum"]
      .mean()
      .sort_values(ascending=False)
      .reset_index()
)

fig3 = px.bar(
    acorn,
    x="Acorn_grouped",
    y="energy_sum",
    text_auto=".2f",
    title="Average Energy Consumption by ACORN Group"
)

st.plotly_chart(fig3, use_container_width=True)

# -----------------------------
# Weather Correlation
# -----------------------------

st.subheader("Temperature vs Energy")

fig4 = px.scatter(
    df.sample(5000, random_state=42),
    x="temperatureMax",
    y="energy_sum",
    opacity=0.5,
    title="Temperature vs Daily Energy Consumption"
)

st.plotly_chart(fig4, use_container_width=True)

# -----------------------------
# Correlation Matrix
# -----------------------------

st.subheader("Correlation Heatmap")

corr = df[
    [
        "energy_sum",
        "temperatureMax",
        "humidity",
        "windSpeed",
        "pressure",
        "lag_1",
        "lag_7",
        "rolling_mean_7"
    ]
].corr()

fig5 = px.imshow(
    corr,
    text_auto=True,
    aspect="auto",
    color_continuous_scale="RdBu_r"
)

st.plotly_chart(fig5, use_container_width=True)