"""
pages/02_Characteristic_Table.py
=================================
Characteristic Table — Applications, Offers and Acceptances by demographic.
Mirrors the team lead's template.
"""

import streamlit as st
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.filters import apply_sidebar_filters

st.set_page_config(page_title="Characteristic Table", page_icon="📋", layout="wide")

MAROON = "#67072d"

# ── CHECK DATA LOADED ─────────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.warning("Please upload your data on the Home page first.")
    st.stop()

df_full = st.session_state["df"]
df = apply_sidebar_filters(df_full)

st.markdown("## 📋 Characteristic Table")
st.markdown("---")

# ── CHARACTERISTIC DEFINITIONS ────────────────────────────────────────────────
CHARACTERISTICS = {
    "Age band":                                     "age_group_2band",
    "In Care":                                      "in_care_indicator",
    "Caring responsibilities":                      "caring_responsibilities_indicator",
    "Disability":                                   "disability_indicator",
    "Estranged":                                    "estranged_indicator",
    "Ethnic group (high level)":                    "ethnic_group_summary_lvl",
    "Free School Meals":                            "fsm_indicator",
    "Gender":                                       "gender",
    "IMD quintile":                                 "imd_quint_2019",
    "Parental HE Indicator":                        "parental_he_indicator",
    "Region (UK)":                                  "region_uk",
    "School and College - Classification":          "school_classification",
    "School and College - Region":                  "school_region",
    "Served in Armed Forces":                       "served_in_the_armed_forces_indicator",
    "Socio-economic group":                         "sec_group",
    "Tariff (18 YO, 3 A-levels)":                  "tariff_band_18_alevel",
    "TUNDRA quintile":                              "tundra_msoa_quint",
    "Domicile (high level)":                        "app_domicile_high_lvl_3_lvls",
    "Live at home indicator":                       "live_at_home_indicator",
}

# Filter to only characteristics whose columns exist in the data
CHARACTERISTICS = {
    label: col for label, col in CHARACTERISTICS.items()
    if col in df.columns
}

# ── UNKNOWN VALUES TO EXCLUDE ─────────────────────────────────────────────────
EXCLUDE = [
    "Unknown", "Not applicable", "Unanswered", "Not known",
    "Nonclassified", "Don't know", "nan", ""
]

# ── BUILD TABLE ───────────────────────────────────────────────────────────────
rows = []

for char_label, col in CHARACTERISTICS.items():
    grouped = df.groupby(col).agg(
        Apps=("application", "sum"),
        Offers=("offer_as_at_30_june", "sum"),
        Main_Scheme=("acceptance_main_scheme", "sum"),
        Total_Acc=("acceptance", "sum"),
        Declines=("decline_as_at_30_june", "sum"),
        Withdrawals=("withdrawals", "sum"),
    ).reset_index()

    grouped.columns = ["Sub Category", "Apps #", "Offers #",
                       "Main Scheme #", "Total Acc #", "Declines #", "Withdrawals #"]

    # Filter out unknowns
    grouped = grouped[~grouped["Sub Category"].astype(str).isin(EXCLUDE)]

    # Calculate totals for % denominators
    total_apps      = grouped["Apps #"].sum()
    total_offers    = grouped["Offers #"].sum()
    total_main      = grouped["Main Scheme #"].sum()
    total_acc       = grouped["Total Acc #"].sum()
    total_declines  = grouped["Declines #"].sum()
    total_with      = grouped["Withdrawals #"].sum()

    grouped["Apps %"]        = (grouped["Apps #"] / total_apps * 100).round(1) if total_apps > 0 else 0
    grouped["Offers %"]      = (grouped["Offers #"] / total_offers * 100).round(1) if total_offers > 0 else 0
    grouped["Main Scheme %"] = (grouped["Main Scheme #"] / total_main * 100).round(1) if total_main > 0 else 0
    grouped["Total Acc %"]   = (grouped["Total Acc #"] / total_acc * 100).round(1) if total_acc > 0 else 0
    grouped["Clearing #"]    = grouped["Total Acc #"] - grouped["Main Scheme #"]
    grouped["Clearing %"]    = (grouped["Clearing #"] / grouped["Clearing #"].sum() * 100).round(1) if grouped["Clearing #"].sum() > 0 else 0
    grouped["Declines %"]    = (grouped["Declines #"] / total_declines * 100).round(1) if total_declines > 0 else 0
    grouped["Withdrawals %"] = (grouped["Withdrawals #"] / total_with * 100).round(1) if total_with > 0 else 0

    grouped.insert(0, "Characteristic", char_label)

    rows.append(grouped)

if rows:
    table = pd.concat(rows, ignore_index=True)

    # ── COLUMN ORDER ──────────────────────────────────────────────────────────
    col_order = [
        "Characteristic", "Sub Category",
        "Apps #", "Apps %",
        "Offers #", "Offers %",
        "Main Scheme #", "Main Scheme %",
        "Total Acc #", "Total Acc %",
        "Clearing #", "Clearing %",
        "Declines #", "Declines %",
        "Withdrawals #", "Withdrawals %",
    ]
    table = table[[c for c in col_order if c in table.columns]]

    # ── CHARACTERISTIC SELECTOR ───────────────────────────────────────────────
    st.markdown("**Filter characteristics to display:**")
    selected_chars = st.multiselect(
        "Characteristics",
        options=list(CHARACTERISTICS.keys()),
        default=list(CHARACTERISTICS.keys()),
        label_visibility="collapsed"
    )

    if selected_chars:
        table = table[table["Characteristic"].isin(selected_chars)]

    # ── DISPLAY TABLE ─────────────────────────────────────────────────────────
    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        height=600
    )

    # ── DOWNLOAD ──────────────────────────────────────────────────────────────
    csv = table.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download table as CSV",
        data=csv,
        file_name="PERS_Characteristic_Table.csv",
        mime="text/csv"
    )