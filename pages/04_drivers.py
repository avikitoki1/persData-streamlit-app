"""
pages/04_Driver_Analysis.py
===========================
Driver Analysis — Key factors influencing offer rates.
Uses mean difference approach — same as Power BI Key Influencers.
Clean, interpretable, visually compelling.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.filters import apply_sidebar_filters

st.set_page_config(page_title="Driver Analysis", page_icon="🎯", layout="wide")

# ── COLOUR PALETTE ────────────────────────────────────────────────────────────
MAROON   = "#67072d"
TEAL     = "#0d6e7a"
GOLD     = "#c9820a"
GREEN    = "#1a7a4a"
PURPLE   = "#5c2d8a"
CORAL    = "#c94a2d"
NAVY     = "#1a3a6b"

POSITIVE_COLOR = TEAL    # increases offer rate
NEGATIVE_COLOR = CORAL   # decreases offer rate

EXCLUDE = ["Unknown", "Not applicable", "Unanswered", "Not known",
           "Nonclassified", "Don't know", "Not assigned", "nan", ""]

# ── CHECK DATA LOADED ─────────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.warning("Please upload your data on the Home page first.")
    st.stop()

df_full = st.session_state["df"]
df = apply_sidebar_filters(df_full)

st.markdown("## 🎯 Driver Analysis")
st.markdown("*Which characteristics most influence whether an applicant receives an offer?*")
st.markdown("""
> **Method:** For each characteristic category, we calculate the difference between  
> that group's offer rate and the overall offer rate. Positive = above average,  
> Negative = below average. This mirrors the approach used in Power BI Key Influencers.
""")
st.markdown("---")

# ── TARGET SELECTION ──────────────────────────────────────────────────────────
target_option = st.radio(
    "Select outcome to analyse:",
    options=["Offer Rate", "Acceptance Rate", "Application Rate"],
    horizontal=True
)

TARGET_MAP = {
    "Offer Rate":       "offer_as_at_30_june",
    "Acceptance Rate":  "acceptance",
    "Application Rate": "application",
}
TARGET     = TARGET_MAP[target_option]
TARGET_LABEL = target_option

st.markdown("---")

# ── CHARACTERISTICS ───────────────────────────────────────────────────────────
CHARACTERISTICS = {
    "Age group":            "age_group_2band",
    "Ethnicity":            "ethnic_group_summary_lvl",
    "Gender":               "gender",
    "Socioeconomic group":  "sec_group",
    "FSM":                  "fsm_indicator",
    "Parental HE":          "parental_he_indicator",
    "Domicile":             "app_domicile_high_lvl_3_lvls",
    "Disability":           "disability_indicator",
    "In Care":              "in_care_indicator",
    "Estranged":            "estranged_indicator",
    "Priority Subject":     "priority_label",
    "TUNDRA Quintile":      "tundra_msoa_quint",
    "IMD Quintile":         "imd_quint_2019",
    "College":              "college",
    "School":               "school",
}

CHARACTERISTICS = {k: v for k, v in CHARACTERISTICS.items() if v in df.columns}

# ── CALCULATE DRIVERS ─────────────────────────────────────────────────────────
# Overall rate
overall_rate = df[TARGET].mean() * 100

results = []

for char_label, col in CHARACTERISTICS.items():
    grouped = df[~df[col].astype(str).isin(EXCLUDE)].groupby(col).agg(
        Count=(TARGET, "count"),
        Rate=(TARGET, "mean")
    ).reset_index()

    grouped["Rate %"]      = grouped["Rate"] * 100
    grouped["Difference"]  = grouped["Rate %"] - overall_rate
    grouped["Characteristic"] = char_label
    grouped = grouped.rename(columns={col: "Sub Category"})
    grouped["Sub Category"] = grouped["Sub Category"].astype(str)

    results.append(grouped)

all_results = pd.concat(results, ignore_index=True)

# ── TOP DRIVERS CHART ─────────────────────────────────────────────────────────
st.markdown(f"### Overall {TARGET_LABEL}: **{overall_rate:.1f}%**")
st.markdown("---")

# Top 10 positive and negative drivers
top_positive = all_results.nlargest(10, "Difference")
top_negative = all_results.nsmallest(10, "Difference")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### ⬆️ Top Drivers — INCREASE Offer Rate")
    st.caption("*These groups have above-average offer rates*")

    fig_pos = go.Figure()
    fig_pos.add_trace(go.Bar(
        x=top_positive["Difference"],
        y=top_positive["Characteristic"] + " — " + top_positive["Sub Category"],
        orientation="h",
        marker=dict(
            color=top_positive["Difference"],
            colorscale=[[0, "#a8d5ba"], [1, GREEN]],
            showscale=False
        ),
        text=top_positive["Difference"].apply(lambda x: f"+{x:.1f}pp"),
        textposition="outside",
        textfont=dict(color="black", size=11),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Difference from average: +%{x:.1f}pp<br>"
            "<extra></extra>"
        )
    ))
    fig_pos.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(color="black", family="Arial"),
        xaxis=dict(
            tickfont=dict(color="black"),
            title="Percentage points above average",
            gridcolor="#eeeeee"
        ),
        yaxis=dict(tickfont=dict(color="black", size=10)),
        height=450,
        margin=dict(l=10, r=80, t=10, b=10)
    )
    st.plotly_chart(fig_pos, use_container_width=True)

with col2:
    st.markdown("#### ⬇️ Top Drivers — DECREASE Offer Rate")
    st.caption("*These groups have below-average offer rates — key WP concern*")

    fig_neg = go.Figure()
    fig_neg.add_trace(go.Bar(
        x=top_negative["Difference"],
        y=top_negative["Characteristic"] + " — " + top_negative["Sub Category"],
        orientation="h",
        marker=dict(
            color=top_negative["Difference"],
            colorscale=[[0, CORAL], [1, "#f5c4bb"]],
            showscale=False
        ),
        text=top_negative["Difference"].apply(lambda x: f"{x:.1f}pp"),
        textposition="outside",
        textfont=dict(color="black", size=11),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Difference from average: %{x:.1f}pp<br>"
            "<extra></extra>"
        )
    ))
    fig_neg.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(color="black", family="Arial"),
        xaxis=dict(
            tickfont=dict(color="black"),
            title="Percentage points below average",
            gridcolor="#eeeeee"
        ),
        yaxis=dict(tickfont=dict(color="black", size=10)),
        height=450,
        margin=dict(l=10, r=80, t=10, b=10)
    )
    st.plotly_chart(fig_neg, use_container_width=True)

st.markdown("---")

# ── CHARACTERISTIC DEEP DIVE ──────────────────────────────────────────────────
st.markdown("### Deep Dive by Characteristic")
st.markdown("*Select a characteristic to see the full breakdown*")

selected_char = st.selectbox(
    "Choose characteristic",
    options=list(CHARACTERISTICS.keys()),
    label_visibility="collapsed"
)

char_data = all_results[all_results["Characteristic"] == selected_char].copy()
char_data = char_data.sort_values("Rate %", ascending=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"#### {selected_char} — Offer Rate")

    colors = [GREEN if x >= 0 else CORAL for x in char_data["Difference"]]

    fig_char = go.Figure()
    fig_char.add_trace(go.Bar(
        x=char_data["Rate %"],
        y=char_data["Sub Category"],
        orientation="h",
        marker_color=colors,
        text=char_data["Rate %"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside",
        textfont=dict(color="black", size=11),
    ))
    fig_char.add_vline(
        x=overall_rate,
        line_dash="dash",
        line_color=NAVY,
        line_width=2,
        annotation_text=f"Average: {overall_rate:.1f}%",
        annotation_font_color=NAVY,
        annotation_position="top right"
    )
    fig_char.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(color="black", family="Arial"),
        xaxis=dict(
            tickfont=dict(color="black"),
            title=f"{TARGET_LABEL} (%)",
            gridcolor="#eeeeee"
        ),
        yaxis=dict(tickfont=dict(color="black")),
        height=max(300, len(char_data) * 45),
        margin=dict(l=10, r=80, t=10, b=10)
    )
    st.plotly_chart(fig_char, use_container_width=True)

with col2:
    st.markdown(f"#### {selected_char} — Applications Count")

    fig_count = px.bar(
        char_data.sort_values("Count", ascending=True),
        x="Count", y="Sub Category",
        orientation="h",
        color_discrete_sequence=[NAVY],
        text="Count"
    )
    fig_count.update_traces(
        texttemplate="%{text:,}",
        textposition="outside",
        textfont=dict(color="black")
    )
    fig_count.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(color="black", family="Arial"),
        xaxis=dict(
            tickfont=dict(color="black"),
            title="Number of Applications",
            gridcolor="#eeeeee"
        ),
        yaxis=dict(tickfont=dict(color="black")),
        height=max(300, len(char_data) * 45),
        margin=dict(l=10, r=80, t=10, b=10)
    )
    st.plotly_chart(fig_count, use_container_width=True)

st.markdown("---")

# ── FULL RESULTS TABLE ────────────────────────────────────────────────────────
st.markdown("### Full Results Table")

display_df = all_results[[
    "Characteristic", "Sub Category", "Count", "Rate %", "Difference"
]].copy()
display_df["Rate %"]     = display_df["Rate %"].round(1)
display_df["Difference"] = display_df["Difference"].round(1)
display_df = display_df.sort_values("Difference", ascending=False)
display_df.columns = [
    "Characteristic", "Sub Category", "Applications",
    f"{TARGET_LABEL} (%)", "Difference from Average (pp)"
]

st.dataframe(display_df, use_container_width=True, hide_index=True, height=400)

csv = display_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download full driver analysis",
    data=csv,
    file_name="PERS_Driver_Analysis.csv",
    mime="text/csv"
)

st.markdown("---")
st.markdown(f"""
**How to interpret:**
- **Overall {TARGET_LABEL}: {overall_rate:.1f}%** — the baseline for all comparisons
- **Positive difference (green)** → this group has a higher than average {TARGET_LABEL.lower()}
- **Negative difference (red)** → this group has a lower than average {TARGET_LABEL.lower()} — potential equity concern
- **pp** = percentage points difference from the overall average

**Methodology:**
For each characteristic category, the {TARGET_LABEL.lower()} is calculated and compared  
to the overall {TARGET_LABEL.lower()} of {overall_rate:.1f}%. This mirrors the approach  
used in Power BI Key Influencers (Microsoft, 2023) and is consistent with descriptive  
driver analysis in widening participation research (Boliver, 2013; Gorard, 2008).
""")