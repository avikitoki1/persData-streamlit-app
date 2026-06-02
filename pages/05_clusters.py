"""
pages/05_Cluster_Analysis.py
============================
Cluster Analysis — Applicant profile groupings.
K-Means (Lloyd, 1982) + Silhouette Score (Rousseeuw, 1987)

Uses numeric WP indicators only for clustering to avoid encoding issues:
- tundra_msoa_quint (1-5)
- imd_quint_2019 (1-5)
- offer_as_at_30_june (0/1)
- application (0/1)
- acceptance (0/1)

Categorical characteristics used for profiling clusters after assignment.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.filters import apply_sidebar_filters

st.set_page_config(page_title="Cluster Analysis", page_icon="🔵", layout="wide")

MAROON = "#67072d"
CLUSTER_COLORS = ["#67072d", "#9e3a5c", "#d4a0b0", "#6b0f2a",
                  "#c46080", "#f0c8d8", "#4a0520", "#e8899a"]

EXCLUDE = ["Unknown", "Not applicable", "Unanswered", "Not known",
           "Nonclassified", "Don't know", "nan", "", "Not assigned"]

# ── CHECK DATA LOADED ─────────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.warning("Please upload your data on the Home page first.")
    st.stop()

df_full = st.session_state["df"]
df = apply_sidebar_filters(df_full)

st.markdown("## 🔵 Cluster Analysis")
st.markdown("*Grouping applicants by similar deprivation and outcome characteristics*")
st.markdown("""
> **K-Means** — Lloyd (1982): groups applicants into K clusters based on similarity  
> **Silhouette Score** — Rousseeuw (1987): measures cluster quality (-1 to 1, higher = better)  
> **Elbow Method**: helps identify optimal number of clusters
""")
st.markdown("---")

st.markdown("""
**How this works:**  
Applicants are clustered using **numeric WP indicators** (TUNDRA quintile, IMD quintile, 
offer outcome, acceptance outcome). Each cluster is then **profiled** by demographic 
characteristics to reveal distinct applicant groups.
""")

# ── SETTINGS ──────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    n_clusters = st.slider(
        "Number of clusters (K):", 
        min_value=2, max_value=8, value=4
    )

with col2:
    show_elbow = st.checkbox("Show elbow chart to find optimal K", value=False)

# ── PROFILE CHARACTERISTICS ───────────────────────────────────────────────────
st.markdown("**Select characteristics to profile each cluster:**")
PROFILE_COLS = {
    "Age group":            "age_group_2band",
    "Ethnicity":            "ethnic_group_summary_lvl",
    "Gender":               "gender",
    "Socioeconomic group":  "sec_group",
    "FSM":                  "fsm_indicator",
    "Parental HE":          "parental_he_indicator",
    "Domicile":             "app_domicile_high_lvl_3_lvls",
    "Priority Subject":     "priority_label",
    "Disability":           "disability_indicator",
    "College":              "college",
}
PROFILE_COLS = {k: v for k, v in PROFILE_COLS.items() if v in df.columns}

selected_profiles = st.multiselect(
    "Profile characteristics",
    options=list(PROFILE_COLS.keys()),
    default=["Age group", "Ethnicity", "Gender",
             "Socioeconomic group", "FSM", "Parental HE"],
    label_visibility="collapsed"
)

if st.button("▶️ Run Cluster Analysis", type="primary"):
    with st.spinner("Running K-Means clustering..."):

        # ── NUMERIC FEATURES FOR CLUSTERING ───────────────────────────────────
        # Only use numeric columns — avoids encoding issues entirely
        numeric_features = []
        feature_labels   = []

        # Core numeric WP indicators
        candidates = [
            ("tundra_msoa_quint",    "TUNDRA Quintile"),
            ("imd_quint_2019",       "IMD Quintile"),
            ("offer_as_at_30_june",  "Offer Received"),
            ("acceptance",           "Accepted"),
            ("application",          "Applied"),
        ]

        for col, label in candidates:
            if col in df.columns:
                numeric_features.append(col)
                feature_labels.append(label)

        if len(numeric_features) < 2:
            st.error("Not enough numeric features available. Please check your data.")
            st.stop()

        # Prepare clustering dataframe
        cluster_df = df[numeric_features].copy()

        # Convert TUNDRA and IMD string labels to numbers
        tundra_map = {
            "TUNDRA Quintile 1": 1,
            "TUNDRA Quintile 2": 2,
            "TUNDRA Quintile 3": 3,
            "TUNDRA Quintile 4": 4,
            "TUNDRA Quintile 5": 5,
            "Not applicable":    np.nan,
            "Not assigned":      np.nan,
        }

        imd_map = {
            "IMD quintile 1":   1,
            "IMD quintile 2":   2,
            "IMD quintile 3":   3,
            "IMD quintile 4":   4,
            "IMD quintile 5":   5,
            "Not applicable":   np.nan,
            "Not assigned":     np.nan,
        }

        if "tundra_msoa_quint" in cluster_df.columns:
            cluster_df["tundra_msoa_quint"] = cluster_df["tundra_msoa_quint"].map(
                tundra_map
            )

        if "imd_quint_2019" in cluster_df.columns:
            cluster_df["imd_quint_2019"] = cluster_df["imd_quint_2019"].map(
                imd_map
            )



        # Convert all to numeric
        for col in numeric_features:
            cluster_df[col] = pd.to_numeric(cluster_df[col], errors="coerce")

        # Drop rows with NaN in clustering features
        cluster_df = cluster_df.dropna()

        if len(cluster_df) < n_clusters * 10:
            st.warning(f"Not enough data — need at least {n_clusters * 10} rows. "
                      f"Currently have {len(cluster_df):,}. "
                      f"Try reducing K or adjusting filters.")
            st.stop()

        # Scale features — important for K-Means
        scaler  = StandardScaler()
        X_scaled = scaler.fit_transform(cluster_df)

        # ── ELBOW CHART ───────────────────────────────────────────────────────
        if show_elbow:
            wcss    = []
            k_range = range(2, 9)
            for k in k_range:
                km_elbow = KMeans(n_clusters=k, random_state=42, n_init=10)
                km_elbow.fit(X_scaled)
                wcss.append(km_elbow.inertia_)
            elbow_df = pd.DataFrame({"K": list(k_range), "WCSS": wcss})

        # ── K-MEANS ───────────────────────────────────────────────────────────
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = km.fit_predict(X_scaled)

        # Validate clusters
        n_unique = len(set(cluster_labels))
        if n_unique < 2:
            st.warning("K-Means found only 1 cluster — try increasing K or "
                      "adjusting filters to include more data variation.")
            st.stop()

        # Silhouette score
        sil_score = silhouette_score(X_scaled, cluster_labels)

        # Add cluster labels to main df using index alignment
        cluster_df["Cluster"] = [f"Cluster {i+1}" for i in cluster_labels]

        # Add profile columns from original df
        for col in PROFILE_COLS.values():
            if col in df.columns:
                cluster_df[col] = df.loc[cluster_df.index, col]

        # Add outcome columns
        for col in ["offer_as_at_30_june", "acceptance", "acceptance_main_scheme",
                    "decline_as_at_30_june", "withdrawals"]:
            if col in df.columns and col not in cluster_df.columns:
                cluster_df[col] = df.loc[cluster_df.index, col]

    # ── RESULTS ───────────────────────────────────────────────────────────────
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Clusters Found", n_unique)
    with col2:
        st.metric("Silhouette Score", f"{sil_score:.3f}",
                  help="Higher = better cluster separation. >0.5 strong, >0.25 moderate")
        if sil_score > 0.5:
            st.caption("✓ Strong cluster separation")
        elif sil_score > 0.25:
            st.caption("~ Moderate cluster separation")
        else:
            st.caption("⚠ Weak — clusters overlap significantly")
    with col3:
        st.metric("Records clustered", f"{len(cluster_df):,}")

    st.markdown("---")

    if show_elbow:
        st.markdown("#### Elbow Chart — Finding Optimal K")
        st.caption("*Look for the bend — where adding more clusters gives diminishing returns*")
        fig_elbow = px.line(
            elbow_df, x="K", y="WCSS", markers=True,
            color_discrete_sequence=[MAROON]
        )
        fig_elbow.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(color="black"),
            xaxis=dict(tickfont=dict(color="black"), title="Number of Clusters (K)"),
            yaxis=dict(tickfont=dict(color="black"),
                      title="Within-Cluster Sum of Squares"),
            height=300
        )
        st.plotly_chart(fig_elbow, use_container_width=True)
        st.markdown("---")

    # ── CLUSTER SIZE + OFFER RATE ─────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Cluster Sizes")
        sizes = cluster_df["Cluster"].value_counts().reset_index()
        sizes.columns = ["Cluster", "Count"]
        sizes = sizes.sort_values("Cluster")

        fig = px.bar(
            sizes, x="Cluster", y="Count",
            color_discrete_sequence=[MAROON], text="Count"
        )
        fig.update_traces(textposition="outside", textfont=dict(color="black"))
        fig.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(color="black"),
            xaxis=dict(tickfont=dict(color="black")),
            yaxis=dict(tickfont=dict(color="black")),
            height=320
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Offer Rate by Cluster")
        offer_by_cluster = cluster_df.groupby("Cluster").agg(
            Apps=("application", "sum"),
            Offers=("offer_as_at_30_june", "sum")
        ).reset_index()
        offer_by_cluster["Offer Rate %"] = (
            offer_by_cluster["Offers"] / offer_by_cluster["Apps"] * 100
        ).round(1)

        fig2 = px.bar(
            offer_by_cluster, x="Cluster", y="Offer Rate %",
            color_discrete_sequence=[MAROON], text="Offer Rate %"
        )
        fig2.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
            textfont=dict(color="black")
        )
        fig2.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(color="black"),
            xaxis=dict(tickfont=dict(color="black")),
            yaxis=dict(tickfont=dict(color="black")),
            height=320
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # ── CLUSTER PROFILES ──────────────────────────────────────────────────────
    st.markdown("#### Cluster Profiles — Numeric Features")
    st.caption("*Mean values of clustering features per cluster*")

    numeric_profile = cluster_df.groupby("Cluster")[numeric_features].mean().round(2)
    numeric_profile.columns = feature_labels
    st.dataframe(numeric_profile, use_container_width=True)

    st.markdown("---")

    # ── CHARACTERISTIC BREAKDOWN PER CLUSTER ─────────────────────────────────
    if selected_profiles:
        st.markdown("#### Characteristic Breakdown by Cluster")

        for profile_label in selected_profiles:
            col_name = PROFILE_COLS[profile_label]
            if col_name not in cluster_df.columns:
                continue

            st.markdown(f"**{profile_label}**")

            # Filter out unknowns
            profile_data = cluster_df[
                ~cluster_df[col_name].astype(str).isin(EXCLUDE)
            ].copy()

            if len(profile_data) == 0:
                continue

            # Calculate % within each cluster
            cross = pd.crosstab(
                profile_data["Cluster"],
                profile_data[col_name],
                normalize="index"
            ).mul(100).round(1)

            fig_profile = px.bar(
                cross.reset_index().melt(
                    id_vars="Cluster",
                    var_name=profile_label,
                    value_name="Percentage"
                ),
                x="Cluster", y="Percentage",
                color=profile_label,
                barmode="stack",
                color_discrete_sequence=CLUSTER_COLORS,
                text="Percentage"
            )
            fig_profile.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="inside",
                textfont=dict(color="white", size=10)
            )
            fig_profile.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(color="black"),
                xaxis=dict(tickfont=dict(color="black")),
                yaxis=dict(tickfont=dict(color="black"), title="Percentage (%)"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                height=300
            )
            st.plotly_chart(fig_profile, use_container_width=True)

    st.markdown("---")

    # ── DOWNLOAD ──────────────────────────────────────────────────────────────
    csv = cluster_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download cluster assignments as CSV",
        data=csv,
        file_name="PERS_Cluster_Assignments.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.markdown("""
    **References:**
    - Lloyd, S.P. (1982). *Least squares quantization in PCM*. IEEE Transactions
    - Rousseeuw, P.J. (1987). *Silhouettes: A graphical aid to the interpretation 
      and validation of cluster analysis*. Journal of Computational and Applied Mathematics
    """)

else:
    st.info("👆 Choose number of clusters, select profile characteristics, "
            "and click **Run Cluster Analysis** to begin.")