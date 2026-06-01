"""
pages/04_Driver_Analysis.py
===========================
Driver Analysis — Key factors influencing offer rates.
Logistic Regression (Cox, 1958) + Random Forest (Breiman, 2001)

Encoding:
- Random Forest  → Label Encoding (handles non-linear splits, no ordering issue)
- Logistic Regression → One-Hot Encoding (correct odds ratio interpretation)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.filters import apply_sidebar_filters

st.set_page_config(page_title="Driver Analysis", page_icon="🎯", layout="wide")

MAROON = "#67072d"

# ── CHECK DATA LOADED ─────────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.warning("Please upload your data on the Home page first.")
    st.stop()

df_full = st.session_state["df"]
df = apply_sidebar_filters(df_full)

st.markdown("## 🎯 Driver Analysis")
st.markdown("*What factors most influence whether an applicant applies or receives an offer?*")
st.markdown("""
> **Logistic Regression** — Cox (1958): predicts probability of outcome, gives odds ratios  
> **Random Forest** — Breiman (2001): ranks variable importance using Gini impurity  
> **Odds Ratio** — Cornfield (1951): how much each characteristic changes the odds  
> **One-Hot Encoding** for Logistic Regression — avoids implying ordering between categories  
> **Label Encoding** for Random Forest — tree splits handle categories naturally
""")
st.markdown("---")

# ── TARGET SELECTION ──────────────────────────────────────────────────────────
target_option = st.radio(
    "Select outcome to analyse:",
    options=["Offer (offer_as_at_30_june)", "Application (application)"],
    horizontal=True
)
TARGET = "offer_as_at_30_june" if "Offer" in target_option else "application"

# ── FEATURE SELECTION ─────────────────────────────────────────────────────────
FEATURE_COLS = {
    "Age group":            "age_group_2band",
    "Ethnicity":            "ethnic_group_summary_lvl",
    "Socioeconomic group":  "sec_group",
    "FSM":                  "fsm_indicator",
    "Parental HE":          "parental_he_indicator",
    "Gender":               "gender",
    "Domicile":             "app_domicile_high_lvl_3_lvls",
    "Disability":           "disability_indicator",
    "In Care":              "in_care_indicator",
    "Estranged":            "estranged_indicator",
    "Priority Subject":     "priority_label",
    "TUNDRA Quintile":      "tundra_msoa_quint",
    "IMD Quintile":         "imd_quint_2019",
}

FEATURE_COLS = {k: v for k, v in FEATURE_COLS.items() if v in df.columns}

st.markdown("#### Select features to include in the model:")
selected_features = st.multiselect(
    "Features",
    options=list(FEATURE_COLS.keys()),
    default=list(FEATURE_COLS.keys())[:8],
    label_visibility="collapsed"
)

if len(selected_features) < 2:
    st.warning("Please select at least 2 features.")
    st.stop()

if st.button("▶️ Run Driver Analysis", type="primary"):
    with st.spinner("Running analysis — this may take a moment..."):

        # ── PREPARE DATA ──────────────────────────────────────────────────────
        feature_cols = [FEATURE_COLS[f] for f in selected_features]
        model_df = df[[TARGET] + feature_cols].copy()

        # Convert target to numeric
        model_df[TARGET] = pd.to_numeric(model_df[TARGET], errors="coerce")
        model_df = model_df.dropna(subset=[TARGET])
        model_df = model_df[model_df[TARGET].isin([0, 1])]
        model_df[TARGET] = model_df[TARGET].astype(int)

        # Fill NaN in features
        for col in feature_cols:
            try:
                model_df[col] = pd.to_numeric(model_df[col], errors="coerce").fillna(0)
            except Exception:
                model_df[col] = model_df[col].fillna("Unknown").astype(str)

        if len(model_df) < 100:
            st.error("Not enough data after filtering. Please adjust your filters.")
            st.stop()

        y = model_df[TARGET]

        # ── LABEL ENCODING for Random Forest ──────────────────────────────────
        # Trees split on values so label encoding works fine
        # "Young"=0, "Mature"=1 — the split finds the best threshold automatically
        X_rf = pd.DataFrame(index=model_df.index)
        for col in feature_cols:
            if model_df[col].dtype == object:
                le = LabelEncoder()
                X_rf[col] = le.fit_transform(model_df[col].astype(str))
            else:
                X_rf[col] = model_df[col].values

        # ── ONE-HOT ENCODING for Logistic Regression ──────────────────────────
        # Each category gets its own 0/1 column
        # "Black" = [0,1,0,0,0], "White" = [0,0,0,0,1]
        # Odds ratio for each category is then directly interpretable
        # drop_first=True removes reference category to avoid multicollinearity
        cat_cols = [c for c in feature_cols if model_df[c].dtype == object]
        num_cols = [c for c in feature_cols if model_df[c].dtype != object]

        X_lr = pd.get_dummies(
            model_df[feature_cols],
            columns=cat_cols,
            drop_first=True
        ).astype(float)

        # Train/test split — stratify keeps class balance in both sets
        X_rf_train, X_rf_test, y_train, y_test = train_test_split(
            X_rf, y, test_size=0.2, random_state=42, stratify=y
        )

        # Use same indices for logistic regression
        train_idx = X_rf_train.index
        test_idx  = X_rf_test.index
        X_lr_train = X_lr.loc[train_idx]
        X_lr_test  = X_lr.loc[test_idx]

        # ── RANDOM FOREST ─────────────────────────────────────────────────────
        # n_estimators=100 — build 100 decision trees
        # random_state=42 — reproducible results
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_rf_train, y_train)

        importance_df = pd.DataFrame({
            "Feature": selected_features,
            "Importance": rf.feature_importances_
        }).sort_values("Importance", ascending=True)

        rf_score = rf.score(X_rf_test, y_test)

        # ── LOGISTIC REGRESSION ───────────────────────────────────────────────
        # max_iter=1000 — allow enough iterations for convergence
        # C=1.0 (default) — regularisation strength
        lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        lr.fit(X_lr_train, y_train)

        # Odds ratio = e^coefficient
        # OR > 1 → increases odds of outcome
        # OR < 1 → decreases odds of outcome
        coef_df = pd.DataFrame({
            "Feature (One-Hot)": X_lr.columns,
            "Coefficient β": lr.coef_[0],
            "Odds Ratio": np.exp(lr.coef_[0])
        }).sort_values("Odds Ratio", ascending=False)

        lr_score = lr.score(X_lr_test, y_test)

    # ── RESULTS ───────────────────────────────────────────────────────────────
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Random Forest Accuracy", f"{rf_score:.1%}")
    with col2:
        st.metric("Logistic Regression Accuracy", f"{lr_score:.1%}")
    with col3:
        st.metric("Records analysed", f"{len(model_df):,}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Random Forest — Feature Importance")
        st.caption("*Breiman (2001) — reduction in Gini impurity across all trees*")
        st.caption("*Label Encoded: each category mapped to integer for tree splitting*")

        fig = px.bar(
            importance_df, x="Importance", y="Feature",
            orientation="h", color_discrete_sequence=[MAROON],
            text="Importance"
        )
        fig.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside",
            textfont=dict(color="black")
        )
        fig.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(color="black"),
            xaxis_title="Feature Importance", yaxis_title="",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Logistic Regression — Odds Ratios")
        st.caption("*Cornfield (1951) — OR > 1 increases odds, OR < 1 decreases odds*")
        st.caption("*One-Hot Encoded: each category vs reference category*")

        coef_display = coef_df.head(20).sort_values("Odds Ratio", ascending=True)
        colors = [MAROON if x >= 1 else "#9e3a5c" for x in coef_display["Odds Ratio"]]

        fig2 = go.Figure(go.Bar(
            x=coef_display["Odds Ratio"],
            y=coef_display["Feature (One-Hot)"],
            orientation="h",
            marker_color=colors,
            text=coef_display["Odds Ratio"].round(3)
        ))
        fig2.add_vline(x=1, line_dash="dash", line_color="black", line_width=1)
        fig2.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside",
            textfont=dict(color="black")
        )
        fig2.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(color="black"),
            xaxis_title="Odds Ratio", yaxis_title="",
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Random Forest — Full Importance Table")
        importance_display = importance_df.sort_values("Importance", ascending=False).copy()
        importance_display["Importance"] = importance_display["Importance"].round(4)
        st.dataframe(importance_display, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("#### Logistic Regression — Full Odds Ratio Table")
        coef_display_full = coef_df.copy()
        coef_display_full["Coefficient β"] = coef_display_full["Coefficient β"].round(4)
        coef_display_full["Odds Ratio"] = coef_display_full["Odds Ratio"].round(3)
        coef_display_full["Effect"] = coef_display_full["Odds Ratio"].apply(
            lambda x: "↑ Increases odds" if x > 1 else "↓ Decreases odds"
        )
        st.dataframe(coef_display_full, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Download
    csv_rf = importance_df.sort_values("Importance", ascending=False).to_csv(index=False).encode("utf-8")
    csv_lr = coef_df.to_csv(index=False).encode("utf-8")

    col1, col2 = st.columns(2)
    with col1:
        st.download_button("⬇️ Download RF Importance", data=csv_rf,
                           file_name="PERS_RF_Importance.csv", mime="text/csv")
    with col2:
        st.download_button("⬇️ Download LR Odds Ratios", data=csv_lr,
                           file_name="PERS_LR_OddsRatios.csv", mime="text/csv")

    st.markdown("---")
    st.markdown(f"""
    **How to interpret — {target_option}:**
    
    **Random Forest Feature Importance:**
    - Higher value = stronger influence on predicting {TARGET}
    - Values sum to 1.0 across all features
    - Does NOT tell you direction (positive or negative effect)
    
    **Logistic Regression Odds Ratios:**
    - Each row shows one category vs its reference category (dropped by one-hot encoding)
    - OR > 1 → this category **increases** odds of {TARGET}
    - OR < 1 → this category **decreases** odds of {TARGET}
    - OR = 1 → no effect compared to reference
    - Example: `ethnicity_Black OR=0.71` means Black applicants have 29% lower odds than the reference ethnic group
    
    **References:**
    - Cox, D.R. (1958). *The regression analysis of binary sequences*. JRSS Series B
    - Breiman, L. (2001). *Random Forests*. Machine Learning, 45, 5-32
    - Cornfield, J. (1951). *A method of estimating comparative rates from clinical data*
    """)
else:
    st.info("👆 Select your target, choose features, and click **Run Driver Analysis** to begin.")