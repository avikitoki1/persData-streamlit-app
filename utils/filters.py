"""
utils/filters.py
================
Shared sidebar filter logic used across all pages.
"""

import streamlit as st
import pandas as pd


def apply_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Render sidebar filters and return filtered dataframe.
    Call this at the top of every page.
    """
    st.sidebar.markdown("## 🔽 Filters")

    # ── Cycle Year ────────────────────────────────────────────────────────────
    years = sorted(df["cycle_year"].dropna().unique().tolist())
    selected_years = st.sidebar.multiselect(
        "Cycle Year",
        options=years,
        default=[y for y in years if y in [2023, 2024, 2025]]
    )

    # ── College ───────────────────────────────────────────────────────────────
    colleges = sorted(df["college"].dropna().unique().tolist())
    selected_colleges = st.sidebar.multiselect(
        "College",
        options=colleges,
        default=colleges
    )

    # ── School ────────────────────────────────────────────────────────────────
    school_col = "school" if "school" in df.columns else None
    if school_col:
        schools = sorted(df[school_col].dropna().unique().tolist())
        selected_schools = st.sidebar.multiselect(
            "School",
            options=schools,
            default=schools
        )
    else:
        selected_schools = []

    # ── Gender ────────────────────────────────────────────────────────────────
    if "gender" in df.columns:
        genders = sorted(df["gender"].dropna().unique().tolist())
        selected_genders = st.sidebar.multiselect(
            "Gender",
            options=genders,
            default=genders
        )
    else:
        selected_genders = []

    # ── Priority Subject ──────────────────────────────────────────────────────
    priority_col = "priority_label" if "priority_label" in df.columns else None
    if priority_col:
        priorities = sorted(df[priority_col].dropna().unique().tolist())
        selected_priorities = st.sidebar.multiselect(
            "Priority Subject",
            options=priorities,
            default=priorities
        )
    else:
        selected_priorities = []


    #_____Course Name_____________________
    course_col = "course_name" if "course_name" in df.columns else None
    if course_col:
        courses = sorted(df[course_col].dropna().unique().tolist())
        selected_courses = st.sidebar.multiselect(
            "Course Name",
            options=courses,
            default=courses
        )
    else:
        selected_courses = []

    # ── Domicile ──────────────────────────────────────────────────────────────
    dom_col = "app_domicile_high_lvl_3_lvls" if "app_domicile_high_lvl_3_lvls" in df.columns else None
    if dom_col:
        domiciles = sorted(df[dom_col].dropna().unique().tolist())
        selected_domiciles = st.sidebar.multiselect(
            "Domicile",
            options=domiciles,
            default=domiciles
        )
    else:
        selected_domiciles = []

    # ── Placed Status ─────────────────────────────────────────────────────────
    if "placed_status" in df.columns:
        statuses = sorted(df["placed_status"].dropna().unique().tolist())
        selected_statuses = st.sidebar.multiselect(
            "Placed Status",
            options=statuses,
            default=statuses
        )
    else:
        selected_statuses = []

    # ── APPLY FILTERS ─────────────────────────────────────────────────────────
    filtered = df.copy()

    if selected_years:
        filtered = filtered[filtered["cycle_year"].isin(selected_years)]
    if selected_colleges:
        filtered = filtered[filtered["college"].isin(selected_colleges)]
    if school_col and selected_schools:
        filtered = filtered[filtered[school_col].isin(selected_schools)]
    if selected_genders and "gender" in filtered.columns:
        filtered = filtered[filtered["gender"].isin(selected_genders)]
    if priority_col and selected_priorities:
        filtered = filtered[filtered[priority_col].isin(selected_priorities)]
    if dom_col and selected_domiciles:
        filtered = filtered[filtered[dom_col].isin(selected_domiciles)]
    if selected_statuses and "placed_status" in filtered.columns:
        filtered = filtered[filtered["placed_status"].isin(selected_statuses)]

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Showing:** {len(filtered):,} rows")

    return filtered