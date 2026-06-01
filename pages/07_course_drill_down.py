"""
pages/06_Course_Profile.py
==========================
Course Profile — Drill-through analysis for individual courses.
Select a course to see application trends, demographic breakdown,
offer rates and year-on-year comparisons.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.filters import apply_sidebar_filters

st.set_page_config(page_title="Course Profile", page_icon="🎓", layout="wide")

MAROON  = "#67072d"
MAROON2 = "#9e3a5c"
COLORS  = [MAROON, MAROON2, "#d4a0b0", "#6b0f2a", "#c46080",
           "#f0c8d8", "#4a0520", "#e8899a"]

EXCLUDE = ["Unknown", "Not applicable", "Unanswered", "Not known",
           "Nonclassified", "Don't know", "nan", ""]

# ── CHECK DATA LOADED ─────────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.warning("Please upload your data on the Home page first.")
    st.stop()

df_full = st.session_state["df"]

st.markdown("## 🎓 Course Profile")
st.markdown("*Select a course to see its full application and demographic profile*")
st.markdown("---")

# ── COURSE SELECTOR ───────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    # Filter by college first
    colleges = ["All"] + sorted(df_full["college"].dropna().unique().tolist())
    selected_college = st.selectbox("Filter by College", options=colleges)

with col2:
    # Filter by school
    if selected_college != "All":
        school_df = df_full[df_full["college"] == selected_college]
    else:
        school_df = df_full

    schools = ["All"] + sorted(school_df["school"].dropna().unique().tolist())
    selected_school = st.selectbox("Filter by School", options=schools)

with col3:
    # Filter courses based on college and school selection
    if selected_college != "All":
        course_df = df_full[df_full["college"] == selected_college]
    else:
        course_df = df_full

    if selected_school != "All":
        course_df = course_df[course_df["school"] == selected_school]

    courses = sorted(course_df["course_name"].dropna().unique().tolist())
    selected_course = st.selectbox(
        "Select Course",
        options=courses,
        help="Type to search for a course"
    )

st.markdown("---")

# ── FILTER TO SELECTED COURSE ─────────────────────────────────────────────────
df = df_full[df_full["course_name"] == selected_course].copy()

if len(df) == 0:
    st.warning("No data found for this course.")
    st.stop()

# ── COURSE HEADER ─────────────────────────────────────────────────────────────
college = df["college"].mode().iloc[0] if "college" in df.columns else "Unknown"
school  = df["school"].mode().iloc[0] if "school" in df.columns else "Unknown"
subject = df["Z_SUBJCAHGRP1_Name_1"].mode().iloc[0] if "Z_SUBJCAHGRP1_Name_1" in df.columns else "Unknown"

st.markdown(f"### {selected_course}")
st.markdown(f"**College:** {college} &nbsp;|&nbsp; **School:** {school} &nbsp;|&nbsp; **Subject Group:** {subject}")
st.markdown("---")

# ── KPI CARDS ─────────────────────────────────────────────────────────────────
total_apps   = df["application"].sum()
total_offers = df["offer_as_at_30_june"].sum()
total_acc    = df["acceptance"].sum()
offer_rate   = total_offers / total_apps * 100 if total_apps > 0 else 0
acc_rate     = total_acc / total_apps * 100 if total_apps > 0 else 0
years        = sorted(df["cycle_year"].dropna().unique().tolist())

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Applications", f"{total_apps:,}")
with col2:
    st.metric("Total Offers", f"{total_offers:,}")
with col3:
    st.metric("Total Acceptances", f"{total_acc:,}")
with col4:
    st.metric("Offer Rate", f"{offer_rate:.1f}%")
with col5:
    st.metric("Acceptance Rate", f"{acc_rate:.1f}%")

st.markdown("---")

# ── ROW 1: TREND + OFFER RATE TREND ──────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Application Trend by Year")

    yearly = df.groupby("cycle_year").agg(
        Applications=("application", "sum"),
        Offers=("offer_as_at_30_june", "sum"),
        Acceptances=("acceptance", "sum")
    ).reset_index()

    fig = go.Figure()
    for metric, color in [
        ("Applications", MAROON),
        ("Offers", MAROON2),
        ("Acceptances", "#d4a0b0")
    ]:
        fig.add_trace(go.Scatter(
            x=yearly["cycle_year"], y=yearly[metric],
            name=metric, mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=8)
        ))

    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(color="black"),
        xaxis=dict(tickfont=dict(color="black"), title="Cycle Year"),
        yaxis=dict(tickfont=dict(color="black"), title="Count"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=320
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("#### Offer Rate by Year")

    yearly["Offer Rate %"] = (yearly["Offers"] / yearly["Applications"] * 100).round(1)
    yearly["Acceptance Rate %"] = (yearly["Acceptances"] / yearly["Applications"] * 100).round(1)

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=yearly["cycle_year"], y=yearly["Offer Rate %"],
        name="Offer Rate %", marker_color=MAROON,
        text=yearly["Offer Rate %"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside", textfont=dict(color="black")
    ))
    fig2.add_trace(go.Bar(
        x=yearly["cycle_year"], y=yearly["Acceptance Rate %"],
        name="Acceptance Rate %", marker_color=MAROON2,
        text=yearly["Acceptance Rate %"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside", textfont=dict(color="black")
    ))
    fig2.update_layout(
        barmode="group",
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(color="black"),
        xaxis=dict(tickfont=dict(color="black"), title="Cycle Year"),
        yaxis=dict(tickfont=dict(color="black"), title="Rate (%)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=320
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ── ROW 2: GENDER + ETHNICITY ─────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Gender Breakdown")

    if "gender" in df.columns:
        gender_df = df[~df["gender"].astype(str).isin(EXCLUDE)]
        gender_counts = gender_df.groupby("gender")["application"].sum().reset_index()
        gender_counts.columns = ["Gender", "Applications"]
        gender_counts = gender_counts[gender_counts["Applications"] > 0]

        fig3 = px.pie(
            gender_counts, values="Applications", names="Gender",
            color_discrete_sequence=COLORS, hole=0.4
        )
        fig3.update_traces(
            textfont=dict(color="black"),
            textinfo="percent+label"
        )
        fig3.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(color="black"),
            height=320,
            showlegend=True
        )
        st.plotly_chart(fig3, use_container_width=True)

with col2:
    st.markdown("#### Ethnicity Breakdown")

    if "ethnic_group_summary_lvl" in df.columns:
        eth_df = df[~df["ethnic_group_summary_lvl"].astype(str).isin(EXCLUDE)]
        eth_counts = eth_df.groupby("ethnic_group_summary_lvl")["application"].sum().reset_index()
        eth_counts.columns = ["Ethnicity", "Applications"]
        eth_counts = eth_counts[eth_counts["Applications"] > 0]
        eth_counts = eth_counts.sort_values("Applications", ascending=True)

        fig4 = px.bar(
            eth_counts, x="Applications", y="Ethnicity",
            orientation="h", color_discrete_sequence=[MAROON],
            text="Applications"
        )
        fig4.update_traces(
            textposition="outside",
            textfont=dict(color="black")
        )
        fig4.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(color="black"),
            xaxis=dict(tickfont=dict(color="black")),
            yaxis=dict(tickfont=dict(color="black")),
            height=320
        )
        st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# ── ROW 3: AGE + SEC ──────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Age Group Breakdown")

    if "age_group_2band" in df.columns:
        age_df = df[~df["age_group_2band"].astype(str).isin(EXCLUDE)]
        age_counts = age_df.groupby("age_group_2band")["application"].sum().reset_index()
        age_counts.columns = ["Age Group", "Applications"]

        fig5 = px.pie(
            age_counts, values="Applications", names="Age Group",
            color_discrete_sequence=COLORS, hole=0.4
        )
        fig5.update_traces(
            textfont=dict(color="black"),
            textinfo="percent+label"
        )
        fig5.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(color="black"),
            height=320
        )
        st.plotly_chart(fig5, use_container_width=True)

with col2:
    st.markdown("#### Socioeconomic Group Breakdown")

    if "sec_group" in df.columns:
        sec_df = df[~df["sec_group"].astype(str).isin(EXCLUDE)]
        sec_counts = sec_df.groupby("sec_group")["application"].sum().reset_index()
        sec_counts.columns = ["SEC Group", "Applications"]
        sec_counts = sec_counts[sec_counts["Applications"] > 0]
        sec_counts = sec_counts.sort_values("Applications", ascending=True)

        fig6 = px.bar(
            sec_counts, x="Applications", y="SEC Group",
            orientation="h", color_discrete_sequence=[MAROON],
            text="Applications"
        )
        fig6.update_traces(
            textposition="outside",
            textfont=dict(color="black")
        )
        fig6.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(color="black"),
            xaxis=dict(tickfont=dict(color="black")),
            yaxis=dict(tickfont=dict(color="black")),
            height=320
        )
        st.plotly_chart(fig6, use_container_width=True)

st.markdown("---")

# ── ROW 4: TUNDRA + DOMICILE ──────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### TUNDRA Quintile Distribution")
    st.caption("*Quintile 1 = most deprived areas*")

    if "tundra_msoa_quint" in df.columns:
        tundra_df = df[df["tundra_msoa_quint"].notna()]
        tundra_df = tundra_df[pd.to_numeric(
            tundra_df["tundra_msoa_quint"], errors="coerce").notna()]
        tundra_counts = tundra_df.groupby(
            "tundra_msoa_quint")["application"].sum().reset_index()
        tundra_counts.columns = ["TUNDRA Quintile", "Applications"]
        tundra_counts["TUNDRA Quintile"] = tundra_counts[
            "TUNDRA Quintile"].astype(str)

        fig7 = px.bar(
            tundra_counts, x="TUNDRA Quintile", y="Applications",
            color_discrete_sequence=[MAROON], text="Applications"
        )
        fig7.update_traces(
            textposition="outside",
            textfont=dict(color="black")
        )
        fig7.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(color="black"),
            xaxis=dict(tickfont=dict(color="black"), title="TUNDRA Quintile"),
            yaxis=dict(tickfont=dict(color="black"), title="Applications"),
            height=320
        )
        st.plotly_chart(fig7, use_container_width=True)

with col2:
    st.markdown("#### Domicile Breakdown")

    if "app_domicile_high_lvl_3_lvls" in df.columns:
        dom_df = df[~df["app_domicile_high_lvl_3_lvls"].astype(str).isin(EXCLUDE)]
        dom_counts = dom_df.groupby(
            "app_domicile_high_lvl_3_lvls")["application"].sum().reset_index()
        dom_counts.columns = ["Domicile", "Applications"]

        fig8 = px.pie(
            dom_counts, values="Applications", names="Domicile",
            color_discrete_sequence=COLORS, hole=0.4
        )
        fig8.update_traces(
            textfont=dict(color="black"),
            textinfo="percent+label"
        )
        fig8.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(color="black"),
            height=320
        )
        st.plotly_chart(fig8, use_container_width=True)

st.markdown("---")

# ── YEAR ON YEAR TABLE ────────────────────────────────────────────────────────
st.markdown("#### Year on Year Summary Table")

yearly["Offer Rate %"]      = yearly["Offer Rate %"].round(1)
yearly["Acceptance Rate %"] = yearly["Acceptance Rate %"].round(1)
yearly = yearly.rename(columns={"cycle_year": "Cycle Year"})

st.dataframe(yearly, use_container_width=True, hide_index=True)

# Download
csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download course data as CSV",
    data=csv,
    file_name=f"PERS_{selected_course.replace(' ', '_')}.csv",
    mime="text/csv"
)