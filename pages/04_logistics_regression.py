"""
pages/05_Logistic_Regression.py
================================
Adjusted Driver Analysis — Logistic Regression with Average Marginal Effects
Answers: which factors genuinely influence outcomes after controlling for everything else?
Consistent with existing app structure, colour palette, and sidebar filters.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.filters import apply_sidebar_filters

st.set_page_config(page_title="Adjusted Driver Analysis", page_icon="📐", layout="wide")

# ── COLOUR PALETTE (matches existing app) ─────────────────────────────────────
MAROON   = "#67072d"
TEAL     = "#0d6e7a"
GOLD     = "#c9820a"
GREEN    = "#1a7a4a"
PURPLE   = "#5c2d8a"
CORAL    = "#c94a2d"
NAVY     = "#1a3a6b"

POSITIVE_COLOR = TEAL
NEGATIVE_COLOR = CORAL

EXCLUDE = ["Unknown", "Not applicable", "Unanswered", "Not known",
           "Nonclassified", "Don't know", "Not assigned", "nan", ""]

# ── CHECK DATA LOADED ─────────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.warning("Please upload your data on the Home page first.")
    st.stop()

df_full = st.session_state["df"]
df = apply_sidebar_filters(df_full)

# ── PAGE HEADER ───────────────────────────────────────────────────────────────
st.markdown("## 📐 Adjusted Driver Analysis")
st.markdown("*Which factors genuinely influence outcomes after controlling for all others simultaneously?*")

with st.expander("ℹ️ How is this different from the Driver Analysis page?", expanded=False):
    st.markdown("""
    **Driver Analysis (previous page)** shows raw differences — each group's outcome rate 
    compared to the overall average, one characteristic at a time. This is descriptive 
    and useful, but variables are **confounded**: for example, the apparent gap for Black 
    applicants may partly reflect the courses they apply to, not ethnicity itself.

    **This page** fits a **logistic regression model** that includes all characteristics 
    simultaneously. It then calculates **Average Marginal Effects (AME)** — the estimated 
    impact of each category on the outcome probability, *holding all other variables 
    constant*. This is the standard approach in widening participation research 
    (Boliver, 2013; Gorard, 2008) and gives a much more defensible basis for 
    identifying genuine equity gaps.

    **Confidence intervals** (95%) are shown for each estimate. Where the interval 
    crosses zero, the effect is not statistically distinguishable from no effect.
    
    **Important caveat:** This is an observational study. Marginal effects indicate 
    association, not proven causation. Unmeasured factors (e.g. personal statement 
    quality, interview performance) are not captured in UCAS data.
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
TARGET       = TARGET_MAP[target_option]
TARGET_LABEL = target_option

st.markdown("---")

# ── CHARACTERISTICS ───────────────────────────────────────────────────────────
ALL_CHARACTERISTICS = {
    "Age group":           "age_group_2band",
    "Ethnicity":           "ethnic_group_summary_lvl",
    "Gender":              "gender",
    "Socioeconomic group": "sec_group",
    "FSM":                 "fsm_indicator",
    "Parental HE":         "parental_he_indicator",
    "Domicile":            "app_domicile_high_lvl_3_lvls",
    "Disability":          "disability_indicator",
    "In Care":             "in_care_indicator",
    "Estranged":           "estranged_indicator",
    "TUNDRA Quintile":     "tundra_msoa_quint",
    "IMD Quintile":        "imd_quint_2019",
    "School":              "school",
    "Priority Subject":    "priority_label",
    "A-Level only":        "qual_alevel_only",
    "BTEC only":           "qual_btec_only",
}

# Keep only columns present in data
CHARACTERISTICS = {k: v for k, v in ALL_CHARACTERISTICS.items() if v in df.columns}

# ── VARIABLE SELECTION UI ─────────────────────────────────────────────────────
st.markdown("### Select characteristics to include in the model")
st.caption("Deselect variables with high missingness or that are not relevant to your question.")

default_vars = [k for k in CHARACTERISTICS if k not in ["School", "Priority Subject"]]
selected_chars = st.multiselect(
    "Characteristics",
    options=list(CHARACTERISTICS.keys()),
    default=default_vars,
    label_visibility="collapsed"
)

if len(selected_chars) < 2:
    st.warning("Please select at least 2 characteristics to run the model.")
    st.stop()

st.markdown("---")

# ── REFERENCE CATEGORY SELECTION ─────────────────────────────────────────────
st.markdown("### Reference categories")
st.caption(
    "Each characteristic needs a reference group. Marginal effects show the difference "
    "compared to this group, after controlling for all other variables. "
    "Choose the group you want to compare others against — typically the largest or most advantaged group."
)

ref_cats = {}
ref_cols = st.columns(min(3, len(selected_chars)))

for i, char_label in enumerate(selected_chars):
    col_name = CHARACTERISTICS[char_label]
    col_data = df[col_name].astype(str)
    valid_cats = sorted([
        c for c in col_data.unique()
        if str(c).strip() not in EXCLUDE and str(c).strip() != "nan" and str(c) != "nan"
    ])
    if not valid_cats:
        continue
    # Default: most frequent non-excluded category
    freq = col_data[~col_data.astype(str).str.strip().isin(EXCLUDE + ["nan"])].value_counts()
    default_ref = str(freq.index[0]) if len(freq) > 0 else valid_cats[0]
    with ref_cols[i % 3]:
        ref_cats[char_label] = st.selectbox(
            f"{char_label} reference",
            options=valid_cats,
            index=valid_cats.index(default_ref) if default_ref in valid_cats else 0,
            key=f"ref_{char_label}"
        )

st.markdown("---")

# ── BUILD MODEL DATASET ───────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def build_model_data(df_hash, target_col, selected_chars_tuple, char_map, ref_cats_frozen, exclude_list):
    """Encode features, drop missing, return X, y, feature names."""

    selected_chars = list(selected_chars_tuple)
    char_map = dict(char_map)
    ref_cats = dict(ref_cats_frozen)
    exclude_list = list(exclude_list)

    model_df = df[[target_col] + [char_map[c] for c in selected_chars]].copy()

    # Replace EXCLUDE values with NaN
    for char_label in selected_chars:
        col = char_map[char_label]
        model_df[col] = model_df[col].astype(str).replace(exclude_list, np.nan)

    # Drop rows with any missing
    n_before = len(model_df)
    model_df = model_df.dropna()
    n_after = len(model_df)
    n_dropped = n_before - n_after

    # One-hot encode, dropping reference categories
    dummies_list = []
    feature_meta = []  # (display_label, category, col_name)

    for char_label in selected_chars:
        col = char_map[char_label]
        ref = ref_cats.get(char_label)
        dummies = pd.get_dummies(model_df[col], prefix=char_label, drop_first=False)
        ref_col = f"{char_label}_{ref}"
        cols_to_keep = [c for c in dummies.columns if c != ref_col]
        dummies_list.append(dummies[cols_to_keep])
        for c in cols_to_keep:
            category = c[len(char_label) + 1:]
            feature_meta.append((char_label, category, c))

    X = pd.concat(dummies_list, axis=1).astype(float)
    y = model_df[target_col].astype(float)

    return X, y, feature_meta, n_dropped, n_after


# Cache key based on dataframe shape + columns (lightweight proxy for content)
df_hash = f"{df.shape}_{TARGET}_{tuple(selected_chars)}_{str(ref_cats)}"

with st.spinner("Fitting logistic regression model..."):
    try:
        X, y, feature_meta, n_dropped, n_used = build_model_data(
            df_hash, TARGET,
            tuple(selected_chars),
            tuple(CHARACTERISTICS.items()),
            tuple(ref_cats.items()),
            tuple(EXCLUDE)
        )
    except Exception as e:
        st.error(f"Error preparing model data: {e}")
        st.stop()

if len(X) < 100:
    st.error("Too few rows after filtering to fit a reliable model. Try selecting fewer characteristics or adjusting sidebar filters.")
    st.stop()

# ── FIT MODEL AND COMPUTE MARGINAL EFFECTS ────────────────────────────────────
@st.cache_data(show_spinner=False)
def fit_and_marginal_effects(X_values, y_values, feature_meta_tuple, n_bootstrap=200):
    """
    Fit logistic regression, compute Average Marginal Effects (AME) via delta method
    and bootstrap confidence intervals.
    
    AME for binary dummy variable: mean over all obs of [P(y=1|x_k=1) - P(y=1|x_k=0)]
    holding all other variables at observed values.
    Bootstrap: refit on resampled data n_bootstrap times.
    """
    X = pd.DataFrame(X_values)
    y = pd.Series(y_values)
    feature_meta = list(feature_meta_tuple)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            C=1.0,
            random_state=42
        ))
    ])
    pipe.fit(X, y)

    def compute_ame(pipe, X):
        ames = []
        X_arr = X.values.copy()
        for i in range(X_arr.shape[1]):
            X1 = X_arr.copy(); X1[:, i] = 1
            X0 = X_arr.copy(); X0[:, i] = 0
            p1 = pipe.predict_proba(X1)[:, 1]
            p0 = pipe.predict_proba(X0)[:, 1]
            ames.append(np.mean(p1 - p0) * 100)  # convert to pp
        return np.array(ames)

    ames = compute_ame(pipe, X)

    # Bootstrap CIs
    boot_ames = []
    rng = np.random.default_rng(42)
    n = len(X)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        X_b = X.iloc[idx]
        y_b = y.iloc[idx]
        if y_b.nunique() < 2:
            continue
        pipe_b = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=500, solver="lbfgs", C=1.0, random_state=0))
        ])
        pipe_b.fit(X_b, y_b)
        boot_ames.append(compute_ame(pipe_b, X))

    boot_arr = np.array(boot_ames)
    ci_low  = np.percentile(boot_arr, 2.5, axis=0)
    ci_high = np.percentile(boot_arr, 97.5, axis=0)

    results = []
    for i, (char_label, category, col_name) in enumerate(feature_meta):
        results.append({
            "Characteristic":      char_label,
            "Category":            category,
            "AME (pp)":            round(ames[i], 2),
            "CI Lower (pp)":       round(ci_low[i], 2),
            "CI Upper (pp)":       round(ci_high[i], 2),
            "Significant":         not (ci_low[i] <= 0 <= ci_high[i]),
        })

    return pd.DataFrame(results)

with st.spinner(f"Computing average marginal effects with bootstrap CIs (this takes ~20–30 seconds)..."):
    try:
        results_df = fit_and_marginal_effects(
            X.values, y.values,
            tuple(map(tuple, feature_meta)),
            n_bootstrap=200
        )
    except Exception as e:
        st.error(f"Model fitting error: {e}")
        st.stop()

# ── SAMPLE SIZE INFO ──────────────────────────────────────────────────────────
col_info1, col_info2, col_info3 = st.columns(3)
col_info1.metric("Rows used in model", f"{n_used:,}")
col_info2.metric("Rows dropped (missing values)", f"{n_dropped:,}")
col_info3.metric("Features in model", f"{len(results_df)}")

if n_dropped / (n_used + n_dropped) > 0.3:
    st.warning(
        f"⚠️ {n_dropped / (n_used + n_dropped):.0%} of rows were dropped due to missing values. "
        "Consider deselecting high-missingness variables (e.g. FSM) or checking data quality."
    )

st.markdown("---")

# ── FOREST PLOT — ALL EFFECTS ─────────────────────────────────────────────────
st.markdown("### Adjusted Marginal Effects — All Variables")
st.caption(
    "Each point shows the estimated effect on outcome probability compared to the reference category, "
    "holding all other variables constant. Bars show 95% confidence intervals. "
    "**Grey = not statistically significant** (CI crosses zero)."
)

plot_df = results_df.sort_values("AME (pp)", ascending=True).copy()
plot_df["Label"] = plot_df["Characteristic"] + " — " + plot_df["Category"]
plot_df["Color"] = plot_df.apply(
    lambda r: (POSITIVE_COLOR if r["AME (pp)"] > 0 else NEGATIVE_COLOR)
    if r["Significant"] else "#aaaaaa",
    axis=1
)

fig_forest = go.Figure()

# Error bars
for _, row in plot_df.iterrows():
    fig_forest.add_trace(go.Scatter(
        x=[row["CI Lower (pp)"], row["CI Upper (pp)"]],
        y=[row["Label"], row["Label"]],
        mode="lines",
        line=dict(color=row["Color"], width=2),
        showlegend=False,
        hoverinfo="skip"
    ))

# Points
fig_forest.add_trace(go.Scatter(
    x=plot_df["AME (pp)"],
    y=plot_df["Label"],
    mode="markers",
    marker=dict(
        color=plot_df["Color"],
        size=9,
        line=dict(color="white", width=1)
    ),
    text=plot_df.apply(
        lambda r: (
            f"<b>{r['Label']}</b><br>"
            f"AME: {r['AME (pp)']:+.1f}pp<br>"
            f"95% CI: [{r['CI Lower (pp)']:+.1f}, {r['CI Upper (pp)']:+.1f}]pp<br>"
            f"{'Statistically significant' if r['Significant'] else 'Not significant'}"
        ), axis=1
    ),
    hovertemplate="%{text}<extra></extra>",
    showlegend=False
))

# Zero line
fig_forest.add_vline(x=0, line_dash="dash", line_color=NAVY, line_width=1.5)

fig_forest.update_layout(
    plot_bgcolor="white", paper_bgcolor="white",
    font=dict(color="black", family="Arial"),
    xaxis=dict(
        title="Average Marginal Effect (percentage points, vs reference category)",
        gridcolor="#eeeeee",
        tickfont=dict(color="black")
    ),
    yaxis=dict(tickfont=dict(color="black", size=10)),
    height=max(500, len(plot_df) * 28),
    margin=dict(l=20, r=40, t=20, b=40)
)

st.plotly_chart(fig_forest, use_container_width=True)

st.markdown("---")

# ── CHARACTERISTIC DEEP DIVE ──────────────────────────────────────────────────
st.markdown("### Deep Dive by Characteristic")
st.caption("Select a characteristic to see adjusted effects for all its categories.")

selected_char_dd = st.selectbox(
    "Characteristic",
    options=sorted(results_df["Characteristic"].unique()),
    label_visibility="collapsed"
)

char_dd_df = results_df[results_df["Characteristic"] == selected_char_dd].copy()
char_dd_df = char_dd_df.sort_values("AME (pp)", ascending=True)

ref_label = ref_cats.get(selected_char_dd, "")

fig_dd = go.Figure()

colors_dd = char_dd_df.apply(
    lambda r: (POSITIVE_COLOR if r["AME (pp)"] > 0 else NEGATIVE_COLOR)
    if r["Significant"] else "#aaaaaa",
    axis=1
)

# CI bars
for _, row in char_dd_df.iterrows():
    color = (POSITIVE_COLOR if row["AME (pp)"] > 0 else NEGATIVE_COLOR) if row["Significant"] else "#aaaaaa"
    fig_dd.add_trace(go.Scatter(
        x=[row["CI Lower (pp)"], row["CI Upper (pp)"]],
        y=[row["Category"], row["Category"]],
        mode="lines",
        line=dict(color=color, width=3),
        showlegend=False,
        hoverinfo="skip"
    ))

# Points
fig_dd.add_trace(go.Scatter(
    x=char_dd_df["AME (pp)"],
    y=char_dd_df["Category"],
    mode="markers+text",
    marker=dict(color=colors_dd.tolist(), size=12, line=dict(color="white", width=1.5)),
    text=char_dd_df["AME (pp)"].apply(lambda x: f"{x:+.1f}pp"),
    textposition="middle right",
    textfont=dict(color="black", size=11),
    hovertemplate=(
        "<b>%{y}</b><br>"
        "AME: %{x:+.1f}pp vs " + ref_label + "<br>"
        "<extra></extra>"
    ),
    showlegend=False
))

fig_dd.add_vline(x=0, line_dash="dash", line_color=NAVY, line_width=1.5,
                 annotation_text=f"Reference: {ref_label}",
                 annotation_font_color=NAVY,
                 annotation_position="top right")

fig_dd.update_layout(
    plot_bgcolor="white", paper_bgcolor="white",
    font=dict(color="black", family="Arial"),
    xaxis=dict(
        title=f"Average Marginal Effect (pp vs {ref_label})",
        gridcolor="#eeeeee",
        tickfont=dict(color="black")
    ),
    yaxis=dict(tickfont=dict(color="black", size=12)),
    height=max(300, len(char_dd_df) * 55),
    margin=dict(l=20, r=100, t=20, b=40)
)

st.plotly_chart(fig_dd, use_container_width=True)

st.markdown("---")

# ── FULL RESULTS TABLE ────────────────────────────────────────────────────────
st.markdown("### Full Results Table")
st.caption("Sorted by effect size. Grey rows are not statistically significant at 95% confidence. ⚠️ = small group (<100 observations in model data) — interpret with caution.")

# Add group N counts from X for low-N flagging
group_counts = X.sum(axis=0).to_dict()
results_df["N in group"] = results_df["Category"].apply(
    lambda cat: int(group_counts.get(
        next((k for k in group_counts if k.endswith("_" + str(cat))), ""), 0
    ))
)

display_df = results_df[[
    "Characteristic", "Category", "AME (pp)", "CI Lower (pp)", "CI Upper (pp)", "Significant", "N in group"
]].copy().sort_values("AME (pp)", ascending=False)

display_df["AME (pp)"] = display_df["AME (pp)"].apply(lambda x: f"{x:+.2f}")
display_df["95% CI"] = display_df.apply(
    lambda r: f"[{r['CI Lower (pp)']:+.2f}, {r['CI Upper (pp)']:+.2f}]", axis=1
)
display_df["Significant"] = display_df["Significant"].map({True: "✓ Yes", False: "✗ No"})
display_df["N in group"] = display_df["N in group"].apply(
    lambda n: f"⚠️ {int(n):,}" if n < 100 else f"{int(n):,}"
)
display_df = display_df[["Characteristic", "Category", "AME (pp)", "95% CI", "Significant", "N in group"]]
display_df.columns = ["Characteristic", "Category", "Adj. Effect (pp)", "95% CI", "Significant", "N in group"]

st.dataframe(display_df, use_container_width=True, hide_index=True, height=400)

csv = results_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download full regression results",
    data=csv,
    file_name="PERS_Logistic_Regression_AME.csv",
    mime="text/csv"
)

st.markdown("---")

# ── METHODOLOGY NOTE ──────────────────────────────────────────────────────────
ref_summary = ", ".join([f"{k}: '{v}'" for k, v in ref_cats.items() if k in selected_chars])
st.markdown(f"""
**How to interpret:**
- **Adjusted effect (pp)** — the estimated difference in outcome probability for this group 
  compared to the reference category, *after holding all other variables constant*
- **Reference categories:** {ref_summary}
- **✓ Significant** — the 95% confidence interval does not cross zero; the effect is 
  statistically distinguishable from no effect
- **✗ Not significant** — the CI crosses zero; the effect may be zero or could go either direction
- **pp** = percentage points

**Methodology:**
Logistic regression with all selected characteristics entered simultaneously. 
Average Marginal Effects (AME) computed by averaging the predicted probability difference 
across all observations (Mood, 2010). 95% confidence intervals via percentile bootstrap 
(200 iterations). Rows with missing values on any selected variable are excluded listwise 
({n_dropped:,} rows excluded, {n_used:,} used).

This approach is consistent with the adjusted analysis used in widening participation 
research (Boliver, 2013, 2016; Gorard, 2008) and produces effects comparable to those 
reported in studies of ethnic and socioeconomic inequalities in UK higher education 
admissions.
""")