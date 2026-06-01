"""
pages/01_Overview.py
====================
Overview page — KPI cards and application funnel trends.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.filters import apply_sidebar_filters

st.set_page_config(page_title="Overview", page_icon="📈", layout="wide")

MAROON = "#67072d"

# ── CHECK DATA LOADED ─────────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.warning("Please upload your data on the Home page first.")
    st.stop()

df_full = st.session_state["df"]

# ── SIDEBAR FILTERS ───────────────────────────────────────────────────────────
df = apply_sidebar_filters(df_full)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown(f"## 📈 Overview")
st.markdown("---")

# ── KPI CARDS ─────────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5, col6 = st.columns(6)

total_apps = len(df[df["application"] == 1])
total_offers  = len(df[df["offer_as_at_30_june"] == 1])
total_acc     = len(df[df["acceptance"] == 1])
main_scheme   = len(df[df["acceptance_main_scheme"] == 1])
clearing      = total_acc - main_scheme
acc_rate      = total_acc / total_apps * 100 if total_apps > 0 else 0

with col1:
    st.metric("Total Applicants", f"{total_apps:,}")
with col2:
    st.metric("Total Offers", f"{total_offers:,}")
with col3:
    st.metric("Total Acceptances", f"{total_acc:,}")
with col4:
    st.metric("Main Scheme", f"{main_scheme:,}")
with col5:
    st.metric("Clearing", f"{clearing:,}")
with col6:
    st.metric("Acceptance Rate", f"{acc_rate:.1f}%")

st.markdown("---")

# ── ROW 1: TREND + FUNNEL ─────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Application Funnel Trend")
    yearly = df.groupby("cycle_year").agg(
        Applications=("application", "sum"),
        Offers=("offer_as_at_30_june", "sum"),
        Acceptances=("acceptance", "sum")
    ).reset_index()

    fig = go.Figure()
    for col_name, color in [
        ("Applications", MAROON),
        ("Offers", "#9e3a5c"),
        ("Acceptances", "#3d262d")
    ]:
        fig.add_trace(go.Scatter(
            x=yearly["cycle_year"], y=yearly[col_name],
            name=col_name, mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=8)
        ))

    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis_title="Cycle Year", yaxis_title="Count",
        height=350
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("#### Applications by Priority Subject")
    if "priority_label" in df.columns:
        priority = df.groupby("priority_label")["application"].sum().reset_index()
        priority = priority.sort_values("application", ascending=True)

        fig2 = px.bar(
            priority, x="application", y="priority_label",
            orientation="h", color_discrete_sequence=[MAROON]
        )
        fig2.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis_title="Applications", yaxis_title="",
            height=350
        )
        st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ── ROW 2: YEAR ON YEAR + ACCEPTANCE RATE BY COLLEGE ─────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Offer Rate by Year")
    yearly["Offer Rate %"] = yearly["Offers"] / yearly["Applications"] * 100
    yearly["Acceptance Rate %"] = yearly["Acceptances"] / yearly["Applications"] * 100

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=yearly["cycle_year"], y=yearly["Offer Rate %"],
        name="Offer Rate %", marker_color=MAROON
    ))
    fig3.add_trace(go.Bar(
        x=yearly["cycle_year"], y=yearly["Acceptance Rate %"],
        name="Acceptance Rate %", marker_color="#9e3a5c"
    ))
    fig3.update_layout(
        barmode="group", plot_bgcolor="white", paper_bgcolor="white",
        xaxis_title="Cycle Year", yaxis_title="Rate (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=350
    )
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    st.markdown("#### Acceptance Rate by College")
    college = df.groupby("college").agg(
        Applications=("application", "sum"),
        Acceptances=("acceptance", "sum")
    ).reset_index()
    college["Acceptance Rate %"] = college["Acceptances"] / college["Applications"] * 100
    college = college.sort_values("Acceptance Rate %", ascending=True)

    fig4 = px.bar(
        college, x="Acceptance Rate %", y="college",
        orientation="h", color_discrete_sequence=[MAROON],
        text="Acceptance Rate %"
    )
    fig4.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig4.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis_title="Acceptance Rate %", yaxis_title="",
        height=350
    )
    st.plotly_chart(fig4, use_container_width=True)