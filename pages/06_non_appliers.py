"""
pages/03b_Non_Applier_Profile.py
=================================
Non-Applier Profile Analysis
Profiles applicants in the UCAS system who did NOT complete an application.
Key WP finding — who is being lost before the application stage?
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.filters import apply_sidebar_filters

st.set_page_config(page_title="Non-Applier Profile", page_icon="🔎", layout="wide")

MAROON   = "#67072d"
MAROON2  = "#9e3a5c"
GREY     = "#6c757d"

# ── CHECK DATA LOADED ─────────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.warning("Please upload your data on the Home page first.")
    st.stop()

df_full = st.session_state["df"]
df = apply_sidebar_filters(df_full)

st.markdown("## 🔎 Non-Applier Profile")
st.markdown("*Who is in the UCAS system but not completing their application to Sheffield Hallam?*")
st.markdown("""
> These applicants represent a **pre-application attrition** group — they were tracked in UCAS  
> but did not submit a formal application. Understanding who they are informs widening  
> participation outreach and pre-application support strategies.
""")
st.markdown("---")

# ── SPLIT DATA ────────────────────────────────────────────────────────────────
applied     = df[df["application"] == 1]
not_applied = df[df["application"] == 0]

# ── KPI CARDS ─────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total in UCAS system", f"{len(df):,}")
with col2:
    st.metric("Applied", f"{len(applied):,}", f"{len(applied)/len(df)*100:.1f}%")
with col3:
    st.metric("Did NOT Apply", f"{len(not_applied):,}", f"{len(not_applied)/len(df)*100:.1f}%")
with col4:
    st.metric("Non-Application Rate", f"{len(not_applied)/len(df)*100:.1f}%")

st.markdown("---")

# ── CHARACTERISTIC DEFINITIONS ────────────────────────────────────────────────
CHARACTERISTICS = {
    "Age group":            "age_group_2band",
    "Ethnicity":            "ethnic_group_summary_lvl",
    "Gender":               "gender",
    "FSM":                  "fsm_indicator",
    "Parental HE":          "parental_he_indicator",
    "Domicile":             "app_domicile_high_lvl_3_lvls",
    "Socioeconomic group":  "sec_group",
    "College":              "college",
    "Priority Subject":     "priority_label",
    "Disability":           "disability_indicator",
    "In Care":              "in_care_indicator",
}

CHARACTERISTICS = {k: v for k, v in CHARACTERISTICS.items() if v in df.columns}

EXCLUDE = ["Unknown", "Not applicable", "Unanswered", "Not known",
           "Nonclassified", "Don't know", "nan", ""]

# ── CHARACTERISTIC SELECTOR ───────────────────────────────────────────────────
st.markdown("#### Select characteristic to compare:")
selected_char = st.selectbox(
    "Characteristic",
    options=list(CHARACTERISTICS.keys()),
    label_visibility="collapsed"
)

col_name = CHARACTERISTICS[selected_char]

# ── BUILD COMPARISON ──────────────────────────────────────────────────────────
def get_pct(df_subset, col):
    counts = df_subset[col].value_counts(normalize=True).mul(100).round(1)
    return counts

applied_pct     = get_pct(applied, col_name)
not_applied_pct = get_pct(not_applied, col_name)

comp = pd.DataFrame({
    "Applied %":     applied_pct,
    "Not Applied %": not_applied_pct,
}).fillna(0)

comp["Difference"] = (comp["Not Applied %"] - comp["Applied %"]).round(1)
comp = comp[~comp.index.astype(str).isin(EXCLUDE)]
comp = comp.sort_values("Difference", ascending=False)
comp.index.name = selected_char

# ── CHARTS ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"#### Applied vs Not Applied — {selected_char}")
    st.caption("*Side by side comparison of % distribution*")

    fig_data = []
    for cat in comp.index:
        fig_data.append({
            "Category": str(cat),
            "Group": "Applied",
            "Percentage": comp.loc[cat, "Applied %"]
        })
        fig_data.append({
            "Category": str(cat),
            "Group": "Not Applied",
            "Percentage": comp.loc[cat, "Not Applied %"]
        })

    fig_df = pd.DataFrame(fig_data)

    fig = px.bar(
        fig_df, x="Percentage", y="Category",
        color="Group", orientation="h", barmode="group",
        color_discrete_map={"Applied": MAROON, "Not Applied": MAROON2},
        text="Percentage"
    )
    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        textfont=dict(color="black")
    )
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(color="black"),
        xaxis_title="Percentage (%)", yaxis_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=max(350, len(comp) * 50)
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown(f"#### Overrepresentation in Non-Appliers")
    st.caption("*Positive = more common in non-appliers, Negative = less common*")

    colors = [MAROON if x > 0 else MAROON2 for x in comp["Difference"]]

    fig2 = go.Figure(go.Bar(
        x=comp["Difference"],
        y=comp.index.astype(str),
        orientation="h",
        marker_color=colors,
        text=comp["Difference"].apply(lambda x: f"+{x:.1f}%" if x > 0 else f"{x:.1f}%")
    ))
    fig2.add_vline(x=0, line_color="black", line_width=1)
    fig2.update_traces(
        textposition="outside",
        textfont=dict(color="black")
    )
    fig2.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(color="black"),
        xaxis_title="Difference (Not Applied % - Applied %)",
        yaxis_title="",
        height=max(350, len(comp) * 50)
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ── FULL COMPARISON TABLE ─────────────────────────────────────────────────────
st.markdown("#### Full Comparison Table")
comp_display = comp.copy()
comp_display.columns = ["Applied %", "Not Applied %", "Difference (pp)"]
comp_display["Interpretation"] = comp_display["Difference (pp)"].apply(
    lambda x: f"↑ Overrepresented in non-appliers by {abs(x):.1f}pp" if x > 2
    else f"↓ Underrepresented in non-appliers by {abs(x):.1f}pp" if x < -2
    else "~ Similar in both groups"
)
st.dataframe(comp_display, use_container_width=True)

st.markdown("---")

# ── ALL CHARACTERISTICS SUMMARY ───────────────────────────────────────────────
st.markdown("#### Summary — Top Overrepresented Groups in Non-Appliers")
st.caption("*Across all characteristics — which groups are most overrepresented in the non-applier group?*")

all_diffs = []
for label, col in CHARACTERISTICS.items():
    if col not in df.columns:
        continue
    a_pct = get_pct(applied, col)
    n_pct = get_pct(not_applied, col)
    comp_all = pd.DataFrame({
        "Applied %": a_pct,
        "Not Applied %": n_pct
    }).fillna(0)
    comp_all["Difference"] = comp_all["Not Applied %"] - comp_all["Applied %"]
    comp_all = comp_all[~comp_all.index.astype(str).isin(EXCLUDE)]
    for cat, row in comp_all.iterrows():
        all_diffs.append({
            "Characteristic": label,
            "Sub Category": str(cat),
            "Applied %": row["Applied %"],
            "Not Applied %": row["Not Applied %"],
            "Difference (pp)": round(row["Difference"], 1)
        })

summary_df = pd.DataFrame(all_diffs)
summary_df = summary_df.sort_values("Difference (pp)", ascending=False)

# Top 10 overrepresented
top10 = summary_df.head(10)
fig3 = px.bar(
    top10,
    x="Difference (pp)",
    y=top10["Characteristic"] + " — " + top10["Sub Category"],
    orientation="h",
    color_discrete_sequence=[MAROON],
    text="Difference (pp)"
)
fig3.update_traces(
    texttemplate="+%{text:.1f}pp",
    textposition="outside",
    textfont=dict(color="black")
)
fig3.update_layout(
    plot_bgcolor="white", paper_bgcolor="white",
    font=dict(color="black"),
    xaxis_title="Overrepresentation (percentage points)",
    yaxis_title="",
    height=400
)
st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")
st.markdown("""
**How to interpret:**
- **Positive difference** → this group is overrepresented among non-appliers — 
  they are more likely to be lost before the application stage
- **Negative difference** → this group is underrepresented — 
  they are more likely to complete their application
- **Policy implication** — groups with high positive differences should be 
  targeted for pre-application outreach and support

**Key finding from Sheffield Hallam data:**
Mature applicants, Black and Asian applicants, those with unknown parental HE status,  
and applicants from Nonclassified socioeconomic backgrounds are disproportionately  
lost before the application stage — suggesting targeted pre-application interventions  
are needed for these groups.
""")

# Download
csv = summary_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download full comparison table",
    data=csv,
    file_name="PERS_NonApplier_Profile.csv",
    mime="text/csv"
)