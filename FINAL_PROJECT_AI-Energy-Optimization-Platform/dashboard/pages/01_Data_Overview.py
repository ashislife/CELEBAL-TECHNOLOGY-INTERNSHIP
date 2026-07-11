import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="Data Overview",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Smart Meter Dataset Overview")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "feature_dataset.csv")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=["day"])

df = load_data()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Records", f"{len(df):,}")

with col2:
    st.metric("Households", df["LCLid"].nunique())

with col3:
    st.metric("Features", len(df.columns))

with col4:
    st.metric(
        "Date Range",
        f"{df['day'].min().date()} → {df['day'].max().date()}"
    )

st.divider()

st.subheader("Dataset Preview")
st.dataframe(df.head(20), use_container_width=True)

st.divider()

st.subheader("Numerical Statistics")
st.dataframe(df.describe(), use_container_width=True)