import os
from datetime import datetime
from io import BytesIO

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Optimization Insights",
    page_icon="💡",
    layout="wide"
)

st.title("💡 AI Energy Optimization Insights")

st.caption(
    "Household benchmarking, energy-efficiency scoring, "
    "saving potential and prioritized recommendations."
)


# =========================================================
# PATH
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


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    required_columns = [
        "LCLid",
        "day",
        "energy_sum",
        "Acorn_grouped",
        "stdorToU",
        "month",
        "is_weekend",
        "temperatureMax",
        "rolling_mean_7"
    ]

    return pd.read_csv(
        DATA_PATH,
        usecols=required_columns,
        parse_dates=["day"],
        nrows=500000,
        low_memory=False
    )


df = load_data()


# =========================================================
# HOUSEHOLD SELECTION
# =========================================================

households = sorted(
    df["LCLid"]
    .dropna()
    .unique()
    .tolist()
)

selected_household = st.selectbox(
    "Select Household",
    households
)

household_df = (
    df[df["LCLid"] == selected_household]
    .sort_values("day")
    .copy()
)

if household_df.empty:
    st.warning("No data is available for this household.")
    st.stop()


# =========================================================
# SAFE HELPER
# =========================================================

def safe_percentage_difference(current, baseline):
    if pd.isna(current) or pd.isna(baseline) or baseline == 0:
        return 0.0

    return ((current - baseline) / baseline) * 100


# =========================================================
# BASIC METRICS
# =========================================================

household_average = household_df["energy_sum"].mean()
overall_average = df["energy_sum"].mean()

acorn_group = household_df["Acorn_grouped"].mode().iloc[0]

similar_households_df = df[
    df["Acorn_grouped"] == acorn_group
]

similar_household_average = (
    similar_households_df["energy_sum"].mean()
)

weekday_average = household_df.loc[
    household_df["is_weekend"] == 0,
    "energy_sum"
].mean()

weekend_average = household_df.loc[
    household_df["is_weekend"] == 1,
    "energy_sum"
].mean()

peak_consumption = household_df["energy_sum"].max()
minimum_consumption = household_df["energy_sum"].min()
median_consumption = household_df["energy_sum"].median()

consumption_std = household_df["energy_sum"].std()

p95_consumption = household_df[
    "energy_sum"
].quantile(0.95)

peak_row_index = household_df[
    "energy_sum"
].idxmax()

peak_date = household_df.loc[
    peak_row_index,
    "day"
]

monthly_usage = (
    household_df
    .groupby("month", as_index=False)["energy_sum"]
    .mean()
)

peak_month_number = int(
    monthly_usage.loc[
        monthly_usage["energy_sum"].idxmax(),
        "month"
    ]
)

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


# =========================================================
# ADVANCED OPTIMIZATION SCORING
# =========================================================

risk_score = 0
score_reasons = []

# 1. Compare with similar ACORN households - maximum 30 points
similar_difference = safe_percentage_difference(
    household_average,
    similar_household_average
)

if similar_difference >= 30:
    risk_score += 30
    score_reasons.append(
        "Consumption is at least 30% above similar households."
    )

elif similar_difference >= 20:
    risk_score += 24
    score_reasons.append(
        "Consumption is at least 20% above similar households."
    )

elif similar_difference >= 10:
    risk_score += 15
    score_reasons.append(
        "Consumption is above similar households."
    )

elif similar_difference > 0:
    risk_score += 7


# 2. Weekend usage - maximum 15 points
weekend_difference = safe_percentage_difference(
    weekend_average,
    weekday_average
)

if weekend_difference >= 25:
    risk_score += 15
    score_reasons.append(
        "Weekend energy usage is substantially higher."
    )

elif weekend_difference >= 15:
    risk_score += 10
    score_reasons.append(
        "Weekend usage is moderately higher."
    )

elif weekend_difference >= 5:
    risk_score += 5


# 3. Consumption variability - maximum 20 points
coefficient_of_variation = (
    consumption_std / household_average
    if household_average > 0
    else 0
)

if coefficient_of_variation >= 0.80:
    risk_score += 20
    score_reasons.append(
        "Energy consumption is highly inconsistent."
    )

elif coefficient_of_variation >= 0.60:
    risk_score += 14
    score_reasons.append(
        "Energy consumption has high variability."
    )

elif coefficient_of_variation >= 0.40:
    risk_score += 8


# 4. Peak spikes - maximum 20 points
peak_ratio = (
    p95_consumption / median_consumption
    if median_consumption > 0
    else 1
)

if peak_ratio >= 3:
    risk_score += 20
    score_reasons.append(
        "Frequent high-consumption spikes were detected."
    )

elif peak_ratio >= 2:
    risk_score += 14
    score_reasons.append(
        "Several significant consumption spikes were detected."
    )

elif peak_ratio >= 1.5:
    risk_score += 7


# 5. Temperature sensitivity - maximum 15 points
temperature_correlation = household_df[
    ["temperatureMax", "energy_sum"]
].corr().iloc[0, 1]

if pd.isna(temperature_correlation):
    temperature_correlation = 0.0

absolute_temperature_correlation = abs(
    temperature_correlation
)

if absolute_temperature_correlation >= 0.60:
    risk_score += 15
    score_reasons.append(
        "Consumption is strongly affected by temperature."
    )

elif absolute_temperature_correlation >= 0.40:
    risk_score += 10
    score_reasons.append(
        "Consumption has moderate temperature sensitivity."
    )

elif absolute_temperature_correlation >= 0.20:
    risk_score += 5


risk_score = min(int(round(risk_score)), 100)
efficiency_score = max(100 - risk_score, 0)


# =========================================================
# RISK LEVEL
# =========================================================

if efficiency_score >= 80:
    efficiency_level = "Excellent"
    priority_level = "Low Priority"

elif efficiency_score >= 65:
    efficiency_level = "Good"
    priority_level = "Moderate Priority"

elif efficiency_score >= 45:
    efficiency_level = "Needs Improvement"
    priority_level = "High Priority"

else:
    efficiency_level = "Inefficient"
    priority_level = "Critical Priority"


# =========================================================
# SAVING POTENTIAL
# =========================================================

segment_excess = max(
    household_average - similar_household_average,
    0
)

segment_saving_percentage = (
    segment_excess / household_average * 100
    if household_average > 0
    else 0
)

peak_reduction_percentage = (
    max(p95_consumption - household_average, 0)
    / p95_consumption
    * 10
    if p95_consumption > 0
    else 0
)

weekend_saving_percentage = (
    min(max(weekend_difference, 0) * 0.15, 5)
)

variability_saving_percentage = (
    min(coefficient_of_variation * 5, 5)
)

saving_percentage = (
    segment_saving_percentage
    + peak_reduction_percentage
    + weekend_saving_percentage
    + variability_saving_percentage
)

saving_percentage = min(
    max(saving_percentage, 0),
    25
)

estimated_daily_saving = (
    household_average
    * saving_percentage
    / 100
)

estimated_monthly_saving = (
    estimated_daily_saving * 30
)

estimated_annual_saving = (
    estimated_daily_saving * 365
)


# =========================================================
# RECOMMENDATION ENGINE
# =========================================================

recommendations = []

if similar_difference >= 10:
    recommendations.append({
        "priority": "High",
        "title": "Reduce consumption relative to similar households",
        "action": (
            f"This household consumes {similar_difference:.1f}% more "
            f"than other {acorn_group} households. Review major "
            "appliances and unnecessary continuous loads."
        )
    })

if weekend_difference >= 5:
    recommendations.append({
        "priority": "High",
        "title": "Control weekend energy demand",
        "action": (
            f"Weekend consumption is {weekend_difference:.1f}% higher "
            "than weekday consumption. Schedule heavy appliances "
            "during off-peak periods and avoid simultaneous operation."
        )
    })

if coefficient_of_variation >= 0.40:
    recommendations.append({
        "priority": "Medium",
        "title": "Reduce consumption variability",
        "action": (
            "Daily consumption changes significantly. Use automated "
            "schedules, energy budgets and smart plugs to stabilize usage."
        )
    })

if peak_ratio >= 1.5:
    recommendations.append({
        "priority": "High",
        "title": "Reduce high-consumption spikes",
        "action": (
            "Avoid operating multiple high-load appliances at the same "
            "time. Distribute appliance usage across different periods."
        )
    })

if temperature_correlation <= -0.40:
    recommendations.append({
        "priority": "Medium",
        "title": "Optimize winter heating",
        "action": (
            "Energy demand increases as temperature falls. Improve "
            "insulation, reduce heat loss and use efficient heating controls."
        )
    })

elif temperature_correlation >= 0.40:
    recommendations.append({
        "priority": "Medium",
        "title": "Optimize cooling usage",
        "action": (
            "Energy consumption increases with temperature. Improve "
            "ventilation and operate cooling appliances efficiently."
        )
    })

recommendations.append({
    "priority": "Low",
    "title": "Reduce standby electricity consumption",
    "action": (
        "Disconnect unused electronics or use smart power strips "
        "to remove unnecessary standby loads."
    )
})

recommendations.append({
    "priority": "Low",
    "title": "Track consumption against weekly baseline",
    "action": (
        "Monitor current consumption against the seven-day average "
        "and investigate sudden increases."
    )
})


priority_order = {
    "High": 1,
    "Medium": 2,
    "Low": 3
}

recommendations = sorted(
    recommendations,
    key=lambda item: priority_order[item["priority"]]
)

top_recommendation = recommendations[0]["action"]


# =========================================================
# EXECUTIVE SUMMARY
# =========================================================

executive_summary = (
    f"Household {selected_household} has an energy-efficiency score "
    f"of {efficiency_score}/100 and is categorized as "
    f"{efficiency_level}. Its average daily consumption is "
    f"{household_average:.2f} kWh, compared with "
    f"{similar_household_average:.2f} kWh for similar "
    f"{acorn_group} households. The estimated saving opportunity is "
    f"{saving_percentage:.1f}%, equivalent to approximately "
    f"{estimated_monthly_saving:.2f} kWh per month. "
    f"The highest-priority action is: {top_recommendation}"
)


# =========================================================
# KPI DISPLAY
# =========================================================

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

kpi1.metric(
    "Efficiency Score",
    f"{efficiency_score}/100"
)

kpi2.metric(
    "Efficiency Level",
    efficiency_level
)

kpi3.metric(
    "Priority",
    priority_level
)

kpi4.metric(
    "Potential Saving",
    f"{saving_percentage:.1f}%"
)

kpi5.metric(
    "Monthly Saving",
    f"{estimated_monthly_saving:.2f} kWh"
)


st.divider()


# =========================================================
# EXECUTIVE AI SUMMARY
# =========================================================

st.subheader("AI Optimization Summary")

st.info(executive_summary)


# =========================================================
# BENCHMARKING
# =========================================================

st.subheader("Household Benchmarking")

benchmark1, benchmark2, benchmark3, benchmark4 = st.columns(4)

benchmark1.metric(
    "Household Average",
    f"{household_average:.2f} kWh"
)

benchmark2.metric(
    "Similar Household Average",
    f"{similar_household_average:.2f} kWh",
    delta=f"{similar_difference:.1f}%"
)

benchmark3.metric(
    "Overall Average",
    f"{overall_average:.2f} kWh"
)

benchmark4.metric(
    "Peak Consumption",
    f"{peak_consumption:.2f} kWh"
)


# =========================================================
# SCORE BREAKDOWN
# =========================================================

st.divider()
st.subheader("Optimization Score Breakdown")

score_breakdown = pd.DataFrame({
    "Factor": [
        "Similar-household consumption",
        "Weekend demand",
        "Usage variability",
        "Consumption spikes",
        "Temperature sensitivity"
    ],
    "Risk Indicator": [
        max(similar_difference, 0),
        max(weekend_difference, 0),
        coefficient_of_variation * 100,
        max((peak_ratio - 1) * 100, 0),
        absolute_temperature_correlation * 100
    ]
})

score_chart = px.bar(
    score_breakdown,
    x="Factor",
    y="Risk Indicator",
    text_auto=".1f",
    title="Energy Optimization Risk Indicators"
)

score_chart.update_layout(
    xaxis_title="",
    yaxis_title="Risk Indicator"
)

st.plotly_chart(
    score_chart,
    use_container_width=True
)


# =========================================================
# RECOMMENDATION DISPLAY
# =========================================================

st.divider()
st.subheader("Prioritized Recommendations")

for index, recommendation in enumerate(
    recommendations,
    start=1
):
    if recommendation["priority"] == "High":
        st.error(
            f"{index}. HIGH PRIORITY - "
            f"{recommendation['title']}\n\n"
            f"{recommendation['action']}"
        )

    elif recommendation["priority"] == "Medium":
        st.warning(
            f"{index}. MEDIUM PRIORITY - "
            f"{recommendation['title']}\n\n"
            f"{recommendation['action']}"
        )

    else:
        st.success(
            f"{index}. LOW PRIORITY - "
            f"{recommendation['title']}\n\n"
            f"{recommendation['action']}"
        )


# =========================================================
# CHARTS
# =========================================================

st.divider()

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    comparison_df = pd.DataFrame({
        "Category": [
            "Selected Household",
            "Similar Households",
            "Overall Average"
        ],
        "Average Consumption": [
            household_average,
            similar_household_average,
            overall_average
        ]
    })

    comparison_chart = px.bar(
        comparison_df,
        x="Category",
        y="Average Consumption",
        text_auto=".2f",
        title="Consumption Benchmark"
    )

    st.plotly_chart(
        comparison_chart,
        use_container_width=True
    )


with chart_col2:
    weekday_weekend_df = pd.DataFrame({
        "Usage Type": ["Weekday", "Weekend"],
        "Average Consumption": [
            weekday_average,
            weekend_average
        ]
    }).dropna()

    week_chart = px.bar(
        weekday_weekend_df,
        x="Usage Type",
        y="Average Consumption",
        text_auto=".2f",
        title="Weekday vs Weekend Consumption"
    )

    st.plotly_chart(
        week_chart,
        use_container_width=True
    )


monthly_usage["Month Name"] = monthly_usage[
    "month"
].map(month_names)

monthly_chart = px.line(
    monthly_usage,
    x="Month Name",
    y="energy_sum",
    markers=True,
    title="Average Monthly Energy Consumption"
)

monthly_chart.update_layout(
    xaxis_title="Month",
    yaxis_title="Average Consumption (kWh)"
)

st.plotly_chart(
    monthly_chart,
    use_container_width=True
)


# =========================================================
# PDF CHART GENERATORS
# =========================================================

def create_benchmark_chart():
    chart_buffer = BytesIO()

    chart_df = pd.DataFrame({
        "Category": [
            "Selected",
            "Similar",
            "Overall"
        ],
        "Consumption": [
            household_average,
            similar_household_average,
            overall_average
        ]
    })

    plt.figure(figsize=(7, 4))
    plt.bar(
        chart_df["Category"],
        chart_df["Consumption"]
    )

    plt.title("Average Energy Consumption Benchmark")
    plt.ylabel("Consumption (kWh)")
    plt.tight_layout()

    plt.savefig(
        chart_buffer,
        format="png",
        dpi=150
    )

    plt.close()

    chart_buffer.seek(0)

    return chart_buffer


def create_monthly_chart():
    chart_buffer = BytesIO()

    plt.figure(figsize=(8, 4))

    plt.plot(
        monthly_usage["Month Name"],
        monthly_usage["energy_sum"],
        marker="o"
    )

    plt.title("Average Monthly Energy Consumption")
    plt.xlabel("Month")
    plt.ylabel("Consumption (kWh)")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        chart_buffer,
        format="png",
        dpi=150
    )

    plt.close()

    chart_buffer.seek(0)

    return chart_buffer


# =========================================================
# PDF GENERATOR
# =========================================================

def generate_pdf_report():
    pdf_buffer = BytesIO()

    document = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=35,
        leftMargin=35,
        topMargin=35,
        bottomMargin=35
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        leading=25,
        spaceAfter=18
    )

    section_style = ParagraphStyle(
        "SectionTitle",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        spaceBefore=12,
        spaceAfter=8
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=9,
        leading=14
    )

    story = []

    story.append(
        Paragraph(
            "AI ENERGY OPTIMIZATION REPORT",
            title_style
        )
    )

    story.append(
        Paragraph(
            f"Household: {selected_household}",
            styles["Heading3"]
        )
    )

    story.append(
        Paragraph(
            f"Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}",
            body_style
        )
    )

    story.append(Spacer(1, 14))

    summary_data = [
        ["Metric", "Value"],
        ["Energy Efficiency Score", f"{efficiency_score}/100"],
        ["Efficiency Level", efficiency_level],
        ["Priority Level", priority_level],
        ["Average Daily Consumption", f"{household_average:.2f} kWh"],
        ["Similar Household Average", f"{similar_household_average:.2f} kWh"],
        ["Overall Average", f"{overall_average:.2f} kWh"],
        ["Peak Consumption", f"{peak_consumption:.2f} kWh"],
        ["Peak Date", str(peak_date.date())],
        ["Peak Month", peak_month],
        ["Potential Saving", f"{saving_percentage:.1f}%"],
        ["Estimated Daily Saving", f"{estimated_daily_saving:.2f} kWh"],
        ["Estimated Monthly Saving", f"{estimated_monthly_saving:.2f} kWh"],
        ["Estimated Annual Saving", f"{estimated_annual_saving:.2f} kWh"]
    ]

    summary_table = Table(
        summary_data,
        colWidths=[2.8 * inch, 3.4 * inch]
    )

    summary_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#1F4E78")
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),
            (
                "BACKGROUND",
                (0, 1),
                (-1, -1),
                colors.HexColor("#F3F6F9")
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                8
            ),
            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                8
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                6
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                6
            )
        ])
    )

    story.append(summary_table)

    story.append(
        Paragraph(
            "Executive Summary",
            section_style
        )
    )

    story.append(
        Paragraph(
            executive_summary,
            body_style
        )
    )

    story.append(
        Paragraph(
            "Consumption Benchmark",
            section_style
        )
    )

    benchmark_buffer = create_benchmark_chart()

    story.append(
        Image(
            benchmark_buffer,
            width=6.2 * inch,
            height=3.4 * inch
        )
    )

    story.append(PageBreak())

    story.append(
        Paragraph(
            "Monthly Consumption Pattern",
            section_style
        )
    )

    monthly_buffer = create_monthly_chart()

    story.append(
        Image(
            monthly_buffer,
            width=6.5 * inch,
            height=3.3 * inch
        )
    )

    story.append(
        Paragraph(
            "Prioritized Recommendations",
            section_style
        )
    )

    recommendation_data = [
        ["Priority", "Recommendation", "Recommended Action"]
    ]

    for recommendation in recommendations:
        recommendation_data.append([
            recommendation["priority"],
            Paragraph(
                recommendation["title"],
                body_style
            ),
            Paragraph(
                recommendation["action"],
                body_style
            )
        ])

    recommendation_table = Table(
        recommendation_data,
        colWidths=[
            0.8 * inch,
            1.8 * inch,
            3.9 * inch
        ],
        repeatRows=1
    )

    recommendation_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#1F4E78")
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP"
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                5
            ),
            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                5
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                6
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                6
            )
        ])
    )

    story.append(recommendation_table)

    story.append(Spacer(1, 20))

    story.append(
        Paragraph(
            "This report was automatically generated by the "
            "AI Energy Optimization Platform.",
            body_style
        )
    )

    document.build(story)

    pdf_buffer.seek(0)

    return pdf_buffer


# =========================================================
# DOWNLOAD SECTION
# =========================================================

st.divider()
st.subheader("Download Report")

report_data = pd.DataFrame({
    "Metric": [
        "Household ID",
        "ACORN Group",
        "Efficiency Score",
        "Efficiency Level",
        "Priority Level",
        "Average Daily Consumption",
        "Similar Household Average",
        "Overall Average",
        "Weekday Average",
        "Weekend Average",
        "Peak Consumption",
        "Peak Date",
        "Peak Month",
        "Potential Saving Percentage",
        "Estimated Daily Saving",
        "Estimated Monthly Saving",
        "Estimated Annual Saving",
        "Top Recommendation"
    ],
    "Value": [
        selected_household,
        acorn_group,
        efficiency_score,
        efficiency_level,
        priority_level,
        round(household_average, 3),
        round(similar_household_average, 3),
        round(overall_average, 3),
        (
            round(weekday_average, 3)
            if pd.notna(weekday_average)
            else "N/A"
        ),
        (
            round(weekend_average, 3)
            if pd.notna(weekend_average)
            else "N/A"
        ),
        round(peak_consumption, 3),
        str(peak_date.date()),
        peak_month,
        round(saving_percentage, 2),
        round(estimated_daily_saving, 3),
        round(estimated_monthly_saving, 3),
        round(estimated_annual_saving, 3),
        top_recommendation
    ]
})

download_col1, download_col2 = st.columns(2)

with download_col1:
    csv_data = report_data.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download CSV Report",
        data=csv_data,
        file_name=(
            f"{selected_household}_optimization_report.csv"
        ),
        mime="text/csv",
        use_container_width=True
    )


with download_col2:
    pdf_report = generate_pdf_report()

    st.download_button(
        label="Download streamlit --versionPDF Report",
        data=pdf_report,
        file_name=(
            f"{selected_household}_energy_optimization_report.pdf"
        ),
        mime="application/pdf",
        use_container_width=True
    )