"""
pages/05_Cluster_Analysis.py
============================
Cluster Analysis — Applicant profile groupings.
K-Means (Lloyd, 1982) + Silhouette Score (Rousseeuw, 1987)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import silhouette_score
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.filters import apply_sidebar_filters

st.set_page_config(page_title="Cluster Analysis", page_icon="🔵", layout="wide")

MAROON = "#67072d"
CLUSTER_COLORS = ["#67072d", "#9e3a5c", "#d4a0b0", "#6b0f2a", "#c46080", "#f0c8d8"]

# ── CHECK DATA LOADED ─────────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.warning("Please upload your data on the Home page first.")
    st.stop()

df_full = st.session_state["df"]
df = apply_sidebar_filters(df_full)

st.markdown("## 🔵 Cluster Analysis")
st.markdown("*Grouping applicants by similar characteristics*")
st.markdown("""
> **K-Means** — Lloyd (1982): groups applicants into K clusters based on similarity  
> **Silhouette Score** — Rousseeuw (1987): measures cluster quality (-1 to 1, higher = better)  
> **Elbow Method**: helps identify optimal number of clusters
""")
st.markdown("---")

# ── FEATURE SELECTION ─────────────────────────────────────────────────────────
CLUSTER_FEATURES = {
    "Age group":            "age_group_2band",
    "Ethnicity":            "ethnic_group_summary_lvl",
    "Socioeconomic group":  "sec_group",
    "FSM":                  "fsm_indicator",
    "Parental HE":          "parental_he_indicator",
    "Gender":               "gender",
    "Domicile":             "app_domicile_high_lvl_3_lvls",
    "Disability":           "disability_indicator",
    "Priority Subject":     "priority_label",
    "TUNDRA Quintile":      "tundra_msoa_quint",
    "IMD Quintile":         "imd_quint_2019",
}

CLUSTER_FEATURES = {k: v for k, v in CLUSTER_FEATURES.items() if v in df.columns}

col1, col2 = st.columns(2)

with col1:
    selected_features = st.multiselect(
        "Select clustering features:",
        options=list(CLUSTER_FEATURES.keys()),
        default=["Age group", "Ethnicity", "Socioeconomic group",
                 "FSM", "Parental HE", "TUNDRA Quintile"][:len(CLUSTER_FEATURES)]
    )

with col2:
    n_clusters = st.slider("Number of clusters (K):", min_value=2, max_value=8, value=4)
    show_elbow = st.checkbox("Show elbow chart to find optimal K", value=False)

if len(selected_features) < 2:
    st.warning("Please select at least 2 features.")
    st.stop()

if st.button("▶️ Run Cluster Analysis", type="primary"):
    with st.spinner("Running K-Means clustering..."):

        # ── PREPARE DATA ──────────────────────────────────────────────────────
        feature_cols = [CLUSTER_FEATURES[f] for f in selected_features]
        cluster_df = df[feature_cols + ["application", "offer_as_at_30_june",
                                         "acceptance"]].copy().dropna()

        # Fill NaN in features before encoding
        for col in feature_cols:
            if cluster_df[col].dtype == object:
                cluster_df[col] = cluster_df[col].fillna("Unknown")
            else:
                cluster_df[col] = pd.to_numeric(
                    cluster_df[col], errors="coerce"
                ).fillna(0)

        # Encode categorical features
        X = pd.DataFrame(index=cluster_df.index)
        for col in feature_cols:
            try:
                X[col] = pd.to_numeric(cluster_df[col], errors="coerce").fillna(0)
            except Exception:
                le = LabelEncoder()
                X[col] = le.fit_transform(
                    cluster_df[col].fillna("Unknown").astype(str)
                )

        # Ensure all numeric before scaling
        X = X.astype(float)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # ── ELBOW CHART ───────────────────────────────────────────────────────
        if show_elbow:
            wcss = []
            k_range = range(2, 9)
            for k in k_range:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                km.fit(X_scaled)
                wcss.append(km.inertia_)

            elbow_df = pd.DataFrame({"K": list(k_range), "WCSS": wcss})

        # ── K-MEANS ───────────────────────────────────────────────────────────
        if len(cluster_df) < n_clusters * 10:
            st.warning(f"Not enough data for {n_clusters} clusters. Need at least {n_clusters * 10} rows. Try reducing K or adjusting filters.")
            st.stop()
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = km.fit_predict(X_scaled)
        # Check number of unique clusters found
        n_unique_clusters = len(set(cluster_labels))

        if n_unique_clusters < 2:
            st.warning(f"K-Means only found {n_unique_clusters} cluster. Try reducing K or selecting more features.")
            st.stop()
        sil_score = silhouette_score(X_scaled, cluster_labels)

        cluster_df = cluster_df.copy()
        cluster_df["Cluster"] = [f"Cluster {i+1}" for i in cluster_labels]

        # ── CLUSTER PROFILES ──────────────────────────────────────────────────
        profile_rows = []
        for cluster in sorted(cluster_df["Cluster"].unique()):
            cdf = cluster_df[cluster_df["Cluster"] == cluster]
            row = {"Cluster": cluster, "Count": len(cdf)}

            for col in feature_cols:
                if cdf[col].dtype == object or str(cdf[col].dtype) == "category":
                    row[col] = cdf[col].mode().iloc[0] if len(cdf[col].mode()) > 0 else "Unknown"
                else:
                    row[col] = round(cdf[col].mean(), 1)

            row["Offer Rate %"] = round(
                cdf["offer_as_at_30_june"].sum() / cdf["application"].sum() * 100, 1
            ) if cdf["application"].sum() > 0 else 0

            row["Acceptance Rate %"] = round(
                cdf["acceptance"].sum() / cdf["application"].sum() * 100, 1
            ) if cdf["application"].sum() > 0 else 0

            profile_rows.append(row)

        profile_df = pd.DataFrame(profile_rows)

    # ── RESULTS ───────────────────────────────────────────────────────────────
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Clusters", n_clusters)
    with col2:
        st.metric("Silhouette Score", f"{sil_score:.3f}")
        if sil_score > 0.5:
            st.caption("✓ Strong cluster separation")
        elif sil_score > 0.25:
            st.caption("~ Moderate cluster separation")
        else:
            st.caption("⚠ Weak cluster separation — try different K or features")
    with col3:
        st.metric("Records clustered", f"{len(cluster_df):,}")

    st.markdown("---")

    if show_elbow:
        st.markdown("#### Elbow Chart — Finding Optimal K")
        st.caption("*Look for the 'elbow' — where the curve bends most sharply*")
        fig_elbow = px.line(
            elbow_df, x="K", y="WCSS", markers=True,
            color_discrete_sequence=[MAROON]
        )
        fig_elbow.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis_title="Number of Clusters (K)",
            yaxis_title="Within-Cluster Sum of Squares",
            height=300
        )
        st.plotly_chart(fig_elbow, use_container_width=True)
        st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Cluster Sizes")
        sizes = cluster_df["Cluster"].value_counts().reset_index()
        sizes.columns = ["Cluster", "Count"]
        fig = px.bar(
            sizes, x="Cluster", y="Count",
            color_discrete_sequence=[MAROON], text="Count"
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Offer Rate by Cluster")
        fig2 = px.bar(
            profile_df, x="Cluster", y="Offer Rate %",
            color_discrete_sequence=[MAROON], text="Offer Rate %"
        )
        fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig2.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            height=350
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Cluster Profiles")
    st.caption("*Most common value shown for categorical features, mean for numeric*")

    display_cols = ["Cluster", "Count", "Offer Rate %", "Acceptance Rate %"] + feature_cols
    display_cols = [c for c in display_cols if c in profile_df.columns]
    st.dataframe(profile_df[display_cols], use_container_width=True, hide_index=True)

    csv = profile_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download cluster profiles as CSV",
        data=csv,
        file_name="PERS_Cluster_Profiles.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.markdown("""
    **References:**
    - Lloyd, S.P. (1982). *Least squares quantization in PCM*. IEEE Transactions
    - Rousseeuw, P.J. (1987). *Silhouettes: A graphical aid to the interpretation and validation of cluster analysis*
    """)

else:
    st.info("👆 Select your features, choose K, and click **Run Cluster Analysis** to begin.")