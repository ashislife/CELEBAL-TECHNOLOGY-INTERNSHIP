import os

import pandas as pd
import plotly.express as px
import streamlit as st


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    
    page_title="AI Energy Optimization Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# PATH
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "feature_dataset.csv"
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data(show_spinner=False)
def load_dashboard_data():
    required_columns = [
        "LCLid",
        "day",
        "energy_sum",
        "Acorn_grouped",
        "temperatureMax",
        "is_weekend",
        "month"
    ]

    return pd.read_csv(
        DATA_PATH,
        usecols=required_columns,
        parse_dates=["day"],
        nrows=300000,
        low_memory=False
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.title("⚡ Energy AI")

    st.caption(
        "Autonomous Energy Optimization Platform"
    )

    st.divider()

    st.success("XGBoost Model Ready")
    st.info("Smart Meter Analytics Active")

    st.divider()

    st.markdown("### Platform Modules")

    st.markdown("📊 Data Overview")
    st.markdown("📈 Usage Patterns")
    st.markdown("⚡ Energy Forecast")
    st.markdown("💡 Optimization Insights")
    st.markdown("📄 PDF Reports")

    st.divider()

    st.caption(
        "Built using Python, XGBoost and Streamlit"
    )


# =========================================================
# HEADER
# =========================================================

st.title("⚡ AI Energy Optimization Platform")

st.info(
    """
### Smart Meter Energy Analytics Dashboard

This platform predicts household energy consumption using XGBoost,
identifies energy usage patterns and provides AI-driven optimization
recommendations to improve energy efficiency.
"""
)
st.subheader(
    "Smart Meter Analytics, Consumption Forecasting "
    "and Energy-Saving Insights"
)
st.write(
    "An AI-powered analytics platform that forecasts household "
    "energy consumption, identifies usage patterns and generates "
    "data-driven optimization recommendations."
)


# =========================================================
# LOAD DATA
# =========================================================

with st.spinner("Loading smart meter analytics..."):
    df = load_dashboard_data()


# =========================================================
# KPI CALCULATIONS
# =========================================================

total_records = len(df)
total_households = df["LCLid"].nunique()
average_consumption = df["energy_sum"].mean()
peak_consumption = df["energy_sum"].max()

date_start = df["day"].min()
date_end = df["day"].max()


# =========================================================
# KPI CARDS
# =========================================================

st.subheader("📊 Platform Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "🏠 Households",
    f"{total_households:,}"
)

c2.metric(
    "📄 Records",
    f"{total_records:,}"
)

c3.metric(
    "⚡ Avg Usage",
    f"{average_consumption:.2f} kWh"
)

c4.metric(
    "📈 Model",
    "XGBoost"
)

# =========================================================
# PLATFORM STATUS
# =========================================================

st.divider()
st.subheader("Platform Status")

status1, status2, status3 = st.columns(3)

with status1:
    with st.container(border=True):
        st.markdown("### ⚡ Forecasting")
        st.success("Operational")
        st.write(
            "XGBoost model predicts household energy consumption."
        )

with status2:
    with st.container(border=True):
        st.markdown("### 📈 Usage Analytics")
        st.success("Operational")
        st.write(
            "Monthly, seasonal, household and weather patterns."
        )

with status3:
    with st.container(border=True):
        st.markdown("### 💡 Optimization")
        st.success("Operational")
        st.write(
            "Efficiency scoring, benchmarking and saving recommendations."
        )


# =========================================================
# TREND ANALYSIS
# =========================================================

st.divider()
st.subheader("Overall Energy Consumption Trend")

daily_trend = (
    df.groupby("day", as_index=False)["energy_sum"]
    .mean()
)

trend_chart = px.line(
    daily_trend,
    x="day",
    y="energy_sum",
    title="Average Daily Energy Consumption"
)

trend_chart.update_layout(
    xaxis_title="Date",
    yaxis_title="Average Consumption (kWh)",
    hovermode="x unified"
)

st.plotly_chart(
    trend_chart,
    use_container_width=True
)


# =========================================================
# BUSINESS INSIGHTS
# =========================================================

st.divider()
st.subheader("Key Business Insights")

weekday_average = df.loc[
    df["is_weekend"] == 0,
    "energy_sum"
].mean()

weekend_average = df.loc[
    df["is_weekend"] == 1,
    "energy_sum"
].mean()

monthly_average = (
    df.groupby("month")["energy_sum"]
    .mean()
)

peak_month_number = int(monthly_average.idxmax())

month_names = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December"
}

peak_month = month_names[peak_month_number]

insight1, insight2, insight3 = st.columns(3)

with insight1:
    with st.container(border=True):
        st.markdown("#### 📅 Data Coverage")
        st.write(
            f"{date_start.date()} to {date_end.date()}"
        )
        st.caption(
            "Historical smart meter and weather records."
        )

with insight2:
    with st.container(border=True):
        st.markdown("#### 🗓 Peak Month")
        st.write(peak_month)
        st.caption(
            "Month with the highest average consumption."
        )

with insight3:
    with st.container(border=True):
        st.markdown("#### 🏠 Weekend Behaviour")

        if weekend_average > weekday_average:
            difference = (
                (weekend_average - weekday_average)
                / weekday_average
            ) * 100

            st.write(
                f"Weekend usage is {difference:.1f}% higher."
            )

        else:
            difference = (
                (weekday_average - weekend_average)
                / weekend_average
            ) * 100

            st.write(
                f"Weekday usage is {difference:.1f}% higher."
            )


# =========================================================
# PROJECT OBJECTIVES
# =========================================================

st.divider()
st.subheader("Project Objectives")

tab1, tab2, tab3 = st.tabs(
    [
        "⚡ Forecast Consumption",
        "📈 Identify Usage Patterns",
        "💡 Optimization Insights"
    ]
)

with tab1:
    st.markdown(
        """
        - Predict household energy consumption using XGBoost.
        - Compare actual and predicted consumption.
        - Analyze household-level forecasting performance.
        - Monitor MAE, RMSE and R² metrics.
        """
    )

with tab2:
    st.markdown(
        """
        - Analyze monthly and seasonal energy trends.
        - Compare weekday and weekend consumption.
        - Analyze household ACORN segments.
        - Measure the relationship between weather and energy usage.
        """
    )

with tab3:
    st.markdown(
        """
        - Generate an energy-efficiency score.
        - Compare households with similar customer segments.
        - Estimate daily, monthly and annual saving opportunities.
        - Generate prioritized optimization recommendations.
        - Export professional PDF and CSV reports.
        """
    )


# =========================================================
# TECHNOLOGY STACK
# =========================================================

st.subheader("🛠 Technology Stack")

st.markdown("""
- **Language:** Python
- **Machine Learning:** XGBoost
- **Dashboard:** Streamlit
- **API:** FastAPI
- **Visualization:** Plotly
- **Data Processing:** Pandas, NumPy
""")

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Developed by Ashish Kumar | AI Energy Optimization Platform | 2026"
)