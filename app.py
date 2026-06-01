"""
app.py
======
PERS Analysis Dashboard — Main Entry Point
Upload your PERS_POWERBI_PivotReady.csv to begin analysis.
"""

import streamlit as st

st.set_page_config(
    page_title="PERS Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── STYLING ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background-color: #67072d;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 28px;
    }
    .main-header p {
        color: #e8c0cc;
        margin: 5px 0 0 0;
        font-size: 14px;
    }
    .stMetric {
        background-color: #fae6ee;
        border-left: 4px solid #67072d;
        padding: 10px;
        border-radius: 4px;
    }
    div[data-testid="metric-container"] {
        background-color: #fae6ee;
        border-left: 4px solid #67072d;
        padding: 10px 15px;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📊 PERS Applicant Analysis Dashboard</h1>
    <p>Sheffield Hallam University — UCAS End of Cycle Data</p>
</div>
""", unsafe_allow_html=True)

# ── FILE UPLOAD ───────────────────────────────────────────────────────────────
st.markdown("### Upload Data")
st.markdown("Upload your `PERS_POWERBI_PivotReady.csv` file to begin.")

uploaded_file = st.file_uploader(
    "Choose CSV file",
    type=["csv", "parquet"],  # ← add parquet
    help="Upload PERS_POWERBI_PivotReady.csv or .parquet"
)

if uploaded_file is None:
    st.info("👆 Please upload your PERS data file to continue.")
    st.markdown("""
    **What this dashboard provides:**
    - 📈 **Overview** — KPI cards and application funnel trends
    - 📋 **Characteristic Table** — Applications, Offers and Acceptances by demographic
    - 🔍 **EDA** — Offer rates by ethnicity, deprivation, socioeconomic group
    - 🎯 **Driver Analysis** — Key factors influencing offer rates
    - 🔵 **Cluster Analysis** — Applicant profile groupings
    
    **Data stays private** — your file is processed in memory only and never stored.
    """)
    st.stop()

# ── LOAD AND CACHE DATA ───────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data...")
def load_data(file):
    import pandas as pd
    if file.name.endswith(".parquet"):
        return pd.read_parquet(file)
    else:
        return pd.read_csv(file, low_memory=False)

df = load_data(uploaded_file)

# Store in session state so all pages can access it
st.session_state["df"] = df
st.session_state["data_loaded"] = True

st.success(f"✓ Data loaded — {len(df):,} rows × {len(df.columns)} columns")

# ── QUICK STATS ───────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Applicants", f"{df['application'].sum():,}")
with col2:
    st.metric("Total Offers", f"{df['offer_as_at_30_june'].sum():,}")
with col3:
    st.metric("Total Acceptances", f"{df['acceptance'].sum():,}")
with col4:
    rate = df['acceptance'].sum() / df['application'].sum() * 100 if df['application'].sum() > 0 else 0
    st.metric("Acceptance Rate", f"{rate:.1f}%")

st.markdown("---")
st.markdown("### 👈 Use the sidebar to navigate between pages")
st.markdown(f"**Cycles available:** {sorted(df['cycle_year'].dropna().unique().tolist())}")
st.markdown(f"**Colleges:** {', '.join(sorted(df['college'].dropna().unique().tolist()))}")