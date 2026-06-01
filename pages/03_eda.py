"""
pages/03_EDA.py
===============
Exploratory Data Analysis — Offer rates by key characteristics.
Principle: Tukey (1977) — understand your data before modelling.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.filters import apply_sidebar_filters

st.set_page_config(page_title="EDA", page_icon="🔍", layout="wide")

MAROON = "#67072d"
COLORS = [MAROON, "#9e3a5c", "#a8687b", "#6b0f2a", "#c46080", "#a47487"]

# ── CHECK DATA LOADED ─────────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.warning("Please upload your data on the Home page first.")
    st.stop()

df_full = st.session_state["df"]
df = apply_sidebar_filters(df_full)

st.markdown("## 🔍 Exploratory Data Analysis")
st.markdown("*Tukey (1977) — understand your data before modelling*")
st.markdown("---")

EXCLUDE = ["Unknown", "Not applicable", "Unanswered", "Not known",
           "Nonclassified", "Don't know", "nan", ""]

def offer_rate_chart(df, col, label, title):
    """Create horizontal bar chart of offer rates by characteristic."""
    if col not in df.columns:
        return None
    grouped = df.groupby(col).agg(
        Apps=("application", "sum"),
        Offers=("offer_as_at_30_june", "sum")
    ).reset_index()
    grouped = grouped[~grouped[col].astype(str).isin(EXCLUDE)]
    grouped["Offer Rate %"] = (grouped["Offers"] / grouped["Apps"] * 100).round(1)
    grouped = grouped.sort_values("Offer Rate %", ascending=True)

    fig = px.bar(
        grouped, x="Offer Rate %", y=col,
        orientation="h", color_discrete_sequence=[MAROON],
        text="Offer Rate %"
    )
    fig.update_traces(texttemplate="%{text:.1f}%",
                       textposition="outside",
                       textfont=dict(color="black", size=12)  # ← fix label colour
                       )
    fig.update_layout(
        plot_bgcolor="white", 
        paper_bgcolor="white",
        yaxis_title="", 
        xaxis_title="Offer Rate (%)",
        height=350, 
        font=dict(color="black"),              # ← fix all text colour
        xaxis=dict(color="black"),
        yaxis=dict(color="black"),
    )
    return fig


# ── ROW 1: ETHNICITY + SOCIOECONOMIC ─────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Offer Rate by Ethnicity")
    st.caption("*Reference: Boliver (2013) — ethnic minority applicants face lower offer rates*")
    fig = offer_rate_chart(df, "ethnic_group_summary_lvl", "Ethnicity", "Offer Rate by Ethnicity")
    if fig:
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("#### Offer Rate by Socioeconomic Group")
    st.caption("*Reference: Gorard (2008) — socioeconomic barriers in HE access*")
    fig = offer_rate_chart(df, "sec_group", "SEC Group", "Offer Rate by Socioeconomic Group")
    if fig:
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── ROW 2: TUNDRA + FSM ───────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Offer Rate by TUNDRA Quintile")
    st.caption("*Quintile 1 = most deprived areas, Quintile 5 = least deprived*")
    if "tundra_msoa_quint" in df.columns:
        tundra = df.groupby("tundra_msoa_quint").agg(
            Apps=("application", "sum"),
            Offers=("offer_as_at_30_june", "sum")
        ).reset_index()
        tundra["Offer Rate %"] = (tundra["Offers"] / tundra["Apps"] * 100).round(1)
        tundra = tundra.dropna(subset=["tundra_msoa_quint"])
        tundra["tundra_msoa_quint"] = tundra["tundra_msoa_quint"].astype(str)

        fig = px.bar(
            tundra, x="tundra_msoa_quint", y="Offer Rate %",
            color_discrete_sequence=[MAROON], text="Offer Rate %"
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis_title="TUNDRA Quintile", yaxis_title="Offer Rate (%)",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("#### Offer Rate by Free School Meals")
    st.caption("*FSM is a key widening participation indicator*")
    fig = offer_rate_chart(df, "fsm_indicator", "FSM", "Offer Rate by Free School Meals")
    if fig:
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── ROW 3: AGE + DOMICILE ─────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Offer Rate by Age Group")
    fig = offer_rate_chart(df, "age_group_2band", "Age Group", "Offer Rate by Age Group")
    if fig:
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("#### Offer Rate by Domicile")
    fig = offer_rate_chart(df, "app_domicile_high_lvl_3_lvls", "Domicile", "Offer Rate by Domicile")
    if fig:
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── ROW 4: IMD + PARENTAL HE ──────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Offer Rate by IMD Quintile")
    st.caption("*Index of Multiple Deprivation — 1 = most deprived*")
    if "imd_quint_2019" in df.columns:
        imd = df.groupby("imd_quint_2019").agg(
            Apps=("application", "sum"),
            Offers=("offer_as_at_30_june", "sum")
        ).reset_index()
        imd["Offer Rate %"] = (imd["Offers"] / imd["Apps"] * 100).round(1)
        imd = imd.dropna(subset=["imd_quint_2019"])
        imd["imd_quint_2019"] = imd["imd_quint_2019"].astype(str)

        fig = px.bar(
            imd, x="imd_quint_2019", y="Offer Rate %",
            color_discrete_sequence=[MAROON], text="Offer Rate %"
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis_title="IMD Quintile", yaxis_title="Offer Rate (%)",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("#### Offer Rate by Parental HE")
    st.caption("*First in family to attend university*")
    fig = offer_rate_chart(df, "parental_he_indicator", "Parental HE", "Offer Rate by Parental HE")
    if fig:
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── CORRELATION TABLE ─────────────────────────────────────────────────────────
st.markdown("#### Pearson Correlation with Offer Rate")
st.caption("*Pearson (1895) — r measures linear relationship strength*")

numeric_cols = ["tundra_msoa_quint", "imd_quint_2019",
                "application", "offer_as_at_30_june",
                "acceptance", "acceptance_main_scheme"]

available = [c for c in numeric_cols if c in df.columns]
if available:
    # Convert to numeric — coerce errors to NaN
    corr_df = df[available].apply(pd.to_numeric, errors="coerce")
    corr = corr_df.corr()["offer_as_at_30_june"].drop("offer_as_at_30_june")
    corr_df = corr.reset_index()
    corr_df.columns = ["Variable", "Pearson r"]
    corr_df["Strength"] = corr_df["Pearson r"].abs().apply(
        lambda x: "Strong" if x > 0.5 else "Moderate" if x > 0.3 else "Weak"
    )
    corr_df = corr_df.sort_values("Pearson r", ascending=False)
    st.dataframe(corr_df, use_container_width=True, hide_index=True)