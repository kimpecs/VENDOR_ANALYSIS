"""
=============================================================================
  VENDOR ASSESSMENT DASHBOARD  —  Streamlit App  v3
  Run:  streamlit run vendor_app.py
  Requires processed/ folder from load_vendor_data.py
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
from difflib import SequenceMatcher

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Vendor Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS — clean editorial dark-sidebar look
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; font-size: 14px; }

[data-testid="stSidebar"] { background: #111318; border-right: 1px solid #1f2230; }
[data-testid="stSidebar"] * { color: #d4d0c8 !important; }
[data-testid="stSidebar"] .stSlider label { font-size: 13px !important; }

.stApp { background: #111318; }

/* Metric cards */
div[data-testid="metric-container"] {
    background: #1a1814; border: 1px solid #1f2230;
    border-radius: 10px; padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
div[data-testid="metric-container"] label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 10px !important; letter-spacing: 0.1em !important;
    text-transform: uppercase !important; color: #f4f2ed !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.8rem !important; color: #f4f2ed !important; font-weight: 600;
}

h1, h2 { font-family: 'Syne', sans-serif !important; color: #f4f2ed !important; font-weight: 700; }
h3 { font-family: 'Syne', sans-serif !important; color: #FFD700 !important; font-weight: 600; }

.chart-desc {
    background: #1a1814; border-left: 3px solid #f4f2ed;
    border-radius: 0 8px 8px 0; padding: 10px 16px;
    margin: 0 0 18px 0; font-size: 13px; color: #d4d0c8;
    line-height: 1.6;
}
.chart-desc strong { color: #f4f2ed; font-weight: 500; }

.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; text-transform: uppercase;
    letter-spacing: 0.12em; color: #f4f2ed;
    margin-bottom: 6px; display: block;
}

.insight-card {
    background: #1a1814; border: 1px solid #1f2230;
    border-radius: 10px; padding: 18px 22px; margin-bottom: 12px;
}
.insight-card .label {
    font-family: 'IBM Plex Mono', monospace; font-size: 10px;
    text-transform: uppercase; letter-spacing: 0.1em;
    color: #f4f2ed; margin-bottom: 6px;
}
.insight-card .value {
    font-family: 'Syne', sans-serif; font-size: 1.5rem;
    color: #f4f2ed; font-weight: 600; margin: 0;
}
.insight-card .sub {
    font-size: 12px; color: #f4f2ed; margin-top: 4px;
}

.context-box {
    background: #1a1814; color: #f4f2ed;
    border-radius: 10px; padding: 16px 22px; margin-bottom: 22px;
}
.context-box h4 {
    font-family: 'Syne', sans-serif; font-size: 1rem;
    color: #f4f2ed !important; margin: 0 0 4px 0; font-weight: 600;
}
.context-box p {
    font-family: 'IBM Plex Mono', monospace; font-size: 11px;
    color: #d4d0c8; margin: 0; letter-spacing: 0.05em;
}

.stTabs [data-baseweb="tab-list"] {
    background: #1a1814; border-radius: 10px;
    padding: 4px; border: 1px solid #1f2230; gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 13px !important; font-weight: 500 !important;
    border-radius: 7px !important; color: #d4d0c8 !important;
    padding: 8px 18px !important;
}
.stTabs [aria-selected="true"] {
    background: #1a1814 !important; color: white !important;
}

#MainMenu, footer { visibility: hidden; }
hr { border-color: #e0ddd6; margin: 24px 0; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "processed"
YEARS    = [2020, 2021, 2022, 2023, 2024, 2025, 2026]

# Performance Tool average lead time in days (from your PO data)
# Used for ETA calculation — update if your actual average differs
DEFAULT_PT_LEAD_DAYS = 90


# ---------------------------------------------------------------------------
# DATA LOADERS
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    d = {}
    files = {
        "master":  "02_sage_master.csv",
        "po":      "01_po_transactions.csv",
        "lead":    "01b_po_lead_time_summary.csv",
        "pt_cat":  "03_CPL_catalogue.csv",
        "metabo":  "04_metabo_prices.csv",
        "ronix":   "05_ronix_prices.csv",
        "combined":"06_combined_vendor_prices.csv",
    }
    for key, fname in files.items():
        f = DATA_DIR / fname
        if f.exists():
            df = pd.read_csv(f, dtype=str)
            # Attempt numeric conversion on obvious number columns
            for col in df.columns:
                if any(x in col.lower() for x in [
                    "revenue","price","volume","frequency","qty","cost",
                    "lead","score","margin","days","amount"
                ]):
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            if "ITEMNO" in df.columns:
                df["ITEMNO"] = df["ITEMNO"].str.strip()
            if "item_number" in df.columns:
                df["item_number"] = df["item_number"].str.strip()
            d[key] = df
    return d


@st.cache_data(show_spinner=False)
def build_scores(_data, w_move, w_prof, w_conv):
    master = _data.get("master")
    lead   = _data.get("lead")
    if master is None:
        return None

    df = master.copy()

    # Use Sub Category Name as the category field
    if "Sub Category Name" in df.columns:
        df["_cat"] = df["Sub Category Name"]
    elif "CATEGORY" in df.columns:
        df["_cat"] = df["CATEGORY"]
    else:
        df["_cat"] = "Unknown"

    # Movement: 3-year sum (2022-2024) — avoids COVID dip + 2025/26 partials
    rev_cols  = [f"total_revenue_{y}"    for y in [2022,2023,2024] if f"total_revenue_{y}"    in df.columns]
    vol_cols  = [f"total_volume_{y}"     for y in [2022,2023,2024] if f"total_volume_{y}"     in df.columns]
    freq_cols = [f"sales_frequency_{y}"  for y in [2022,2023,2024] if f"sales_frequency_{y}"  in df.columns]

    df["_rev3"]  = df[rev_cols].sum(axis=1)  if rev_cols  else 0
    df["_vol3"]  = df[vol_cols].sum(axis=1)  if vol_cols  else 0
    df["_freq3"] = df[freq_cols].sum(axis=1) if freq_cols else 0

    # Revenue trend 2022→2024
    if "total_revenue_2022" in df.columns and "total_revenue_2024" in df.columns:
        df["_trend"] = (df["total_revenue_2024"] - df["total_revenue_2022"]) / (df["total_revenue_2022"].abs() + 1)
    else:
        df["_trend"] = 0

    # Profitability: margin %
    if "lifetime_avg_price" in df.columns and "performance_tool_price" in df.columns:
        df["_margin"] = ((df["lifetime_avg_price"] - df["performance_tool_price"])
                         / (df["lifetime_avg_price"].abs() + 0.01)).clip(-1, 1)
    else:
        df["_margin"] = 0

    # Convenience: inverted lead time
    if lead is not None and "avg_lead_time_days" in lead.columns:
        df = df.merge(lead[["ITEMNO","avg_lead_time_days"]], on="ITEMNO", how="left")
        max_lt = df["avg_lead_time_days"].max() or 1
        df["_conv"] = 1 - (df["avg_lead_time_days"].fillna(max_lt) / max_lt)
    else:
        df["_conv"] = 0.5

    # Normalise
    scaler = MinMaxScaler()
    for col in ["_rev3","_vol3","_freq3","_trend","_margin","_conv"]:
        vals = df[[col]].fillna(0)
        if vals.std().iloc[0] > 0:
            df[f"{col}_n"] = scaler.fit_transform(vals)
        else:
            df[f"{col}_n"] = 0

    df["movement_score"]      = (df["_rev3_n"]*0.4 + df["_vol3_n"]*0.3 + df["_freq3_n"]*0.3).round(4)
    df["profitability_score"] = (df["_margin_n"]*0.6 + df["_trend_n"]*0.4).round(4)
    df["convenience_score"]   = df["_conv_n"].round(4)

    total_w = w_move + w_prof + w_conv
    df["composite_score"] = (
        df["movement_score"]      * (w_move / total_w) +
        df["profitability_score"] * (w_prof / total_w) +
        df["convenience_score"]   * (w_conv / total_w)
    ).round(4)

    df["tier"] = pd.cut(
        df["composite_score"],
        bins=[0, 0.33, 0.66, 1.01],
        labels=["Low","Medium","High"],
        include_lowest=True
    )
    return df


@st.cache_data(show_spinner=False)
def build_predictions(_data):
    master = _data.get("master")
    if master is None:
        return None

    results = []
    for _, row in master.iterrows():
        rev = [pd.to_numeric(row.get(f"total_revenue_{y}", 0), errors="coerce") or 0
               for y in YEARS]
        vol = [pd.to_numeric(row.get(f"total_volume_{y}", 0), errors="coerce") or 0
               for y in YEARS]

        # Need at least 3 non-zero years in 2020-2024 (indices 0-4)
        non_zero = [v for v in rev[:5] if v > 0]
        if len(non_zero) < 3:
            continue

        X = np.array(range(5)).reshape(-1, 1)
        y_train = np.array(rev[:5])

        try:
            model = LinearRegression().fit(X, y_train)
            pred_2026 = max(0, float(model.predict([[6]])[0]))
            slope = float(model.coef_[0])
            trend = "Growing" if slope > 500 else ("Declining" if slope < -500 else "Stable")

            # Current selling price: prefer 2026, fallback 2025, then lifetime
            price_2026 = pd.to_numeric(row.get("avg_price_2026", 0), errors="coerce") or 0
            price_2025 = pd.to_numeric(row.get("avg_price_2025", 0), errors="coerce") or 0
            current_price = price_2026 if price_2026 > 0 else price_2025

            results.append({
                "ITEMNO":            str(row.get("ITEMNO","")).strip(),
                "item_description":  str(row.get("item_description", row.get("DESC",""))),
                "sub_category":      str(row.get("Sub Category Name", row.get("CATEGORY",""))),
                "rev_2022":          rev[2],
                "rev_2023":          rev[3],
                "rev_2024":          rev[4],
                "rev_2025":          rev[5],
                "rev_pred_2026":     round(pred_2026, 2),
                "vol_2024":          vol[4],
                "trend":             trend,
                "slope":             round(slope, 2),
                "current_sell_price":current_price,
                "pt_cost":           pd.to_numeric(row.get("performance_tool_price", 0), errors="coerce") or 0,
            })
        except Exception:
            continue

    return pd.DataFrame(results) if results else None


def get_eta(lead_days=None):
    """Return estimated arrival date from today."""
    days = int(lead_days) if lead_days and not np.isnan(float(lead_days)) else DEFAULT_PT_LEAD_DAYS
    eta = datetime.today() + timedelta(days=days)
    return eta.strftime("%B %d, %Y"), days


def chart_container(title, description, chart_fn):
    """Wrapper: renders description then the chart."""
    st.markdown(f"### {title}")
    st.markdown(f"<div class='chart-desc'>{description}</div>", unsafe_allow_html=True)
    chart_fn()


def plotly_defaults():
    return dict(
        plot_bgcolor="#111318",
        paper_bgcolor="#111318",
        font_family="IBM Plex Sans",
        font_color="#f4f2ed",
        font=dict(family="IBM Plex Sans", color="#f4f2ed", size=14),
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis=dict(color="#d4d0c8", gridcolor="#1f2230"),
        yaxis=dict(color="#d4d0c8", gridcolor="#1f2230"),
        legend=dict(font=dict(color="#d4d0c8")),
    )


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
def sidebar(data):
    st.sidebar.image("LIG Logo RGB-Primary.png", width=140)
    st.sidebar.markdown("""
    <div style='padding:10px 0 18px 0;'>
        <div style='font-family:Syne,sans-serif;font-size:1.25rem;
                    font-weight:700;color:#f4f2ed;'>Vendor Intelligence</div>
        <div style='font-family:"IBM Plex Mono",monospace;font-size:9px;
                    letter-spacing:0.12em;text-transform:uppercase;
                    color:#3a3c44;margin-top:3px;'>Assessment Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.sidebar.radio("Navigate", [
        "Overview",
        "Scoring & Ranking",
        "Predictions & PO Timing",
        "Price Analysis",
        "Item Deep Dive",
        "Vendor Comparison",
    ], label_visibility="collapsed")

    st.sidebar.markdown("<hr style='border-color:#1f2230;margin:16px 0;'>", unsafe_allow_html=True)
    st.sidebar.markdown(
        "<span class='section-label' style='color:#3a3c44 !important;'>"
        "Score Weights</span>", unsafe_allow_html=True
    )
    w_move = st.sidebar.slider("Movement",      10, 70, 40, 5)
    w_prof = st.sidebar.slider("Profitability", 10, 70, 35, 5)
    w_conv = st.sidebar.slider("Convenience",   10, 70, 25, 5)
    total  = w_move + w_prof + w_conv
    color  = "#5cb87a" if total == 100 else "#e87c55"
    st.sidebar.markdown(
        f"<div style='font-family:IBM Plex Mono,monospace;font-size:11px;"
        f"color:{color};margin-top:4px;'>"
        f"Total: {total}% {'OK' if total==100 else 'Warning: should equal 100%'}</div>",
        unsafe_allow_html=True
    )

    st.sidebar.markdown("<hr style='border-color:#1f2230;margin:16px 0;'>", unsafe_allow_html=True)
    st.sidebar.markdown(
        "<span class='section-label' style='color:#3a3c44 !important;'>"
        "Ronix Freight Factor</span>", unsafe_allow_html=True
    )
    ronix_ff = st.sidebar.slider("FOB → Landed multiplier", 1.0, 2.0, 1.30, 0.05,
                                  help="1.30 = 30% uplift on FOB to estimate your landed cost")

    st.sidebar.markdown("<hr style='border-color:#1f2230;margin:16px 0;'>", unsafe_allow_html=True)
    st.sidebar.markdown(
        "<span class='section-label' style='color:#3a3c44 !important;'>"
        "Data Status</span>", unsafe_allow_html=True
    )
    for label, key in [("Sage Master","master"),("PO Transactions","po"),
                        ("PT Catalogue","pt_cat"),("Metabo","metabo"),("Ronix","ronix")]:
        ok = key in data
        st.sidebar.markdown(
            f"<div style='font-family:IBM Plex Mono,monospace;font-size:11px;"
            f"color:{'#5cb87a' if ok else '#e87c55'};padding:2px 0;'>"
            f"{'OK' if ok else 'Missing'} {label}</div>",
            unsafe_allow_html=True
        )

    return page.split("  ", 1)[-1].strip(), w_move, w_prof, w_conv, ronix_ff


# ===========================================================================
# PAGE 1 — OVERVIEW
# ===========================================================================
def page_overview(data, scores):
    st.markdown("## Overview")

    master = data.get("master")
    if master is None:
        st.warning("No Sage master data found. Run load_vendor_data.py first.")
        return

    # Use Sub Category Name where available
    cat_col = "Sub Category Name" if "Sub Category Name" in master.columns else "CATEGORY"

    st.markdown(
        "<div class='context-box'><h4>What you're looking at</h4>"
        "<p>A high-level snapshot of your Performance Tool catalogue performance. "
        "All revenue figures are in JMD (Jamaican Dollars). "
        "Scores are composite indexes — Movement reflects how well items sell, "
        "Profitability reflects margin, Convenience reflects lead time.</p></div>",
        unsafe_allow_html=True
    )

    # KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    total_rev = sum(
        master[f"total_revenue_{y}"].sum()
        for y in YEARS if f"total_revenue_{y}" in master.columns
    )
    c1.metric("Active Items",        f"{len(master):,}")
    c2.metric("Total Revenue (JMD)", f"${total_rev:,.0f}")
    c3.metric("Lifetime Volume",
              f"{master['lifetime_total_volume'].sum():,.0f}"
              if "lifetime_total_volume" in master.columns else "N/A")
    c4.metric("Avg Selling Price",
              f"${master['lifetime_avg_price'].mean():,.2f}"
              if "lifetime_avg_price" in master.columns else "N/A")
    c5.metric("Vendors Assessed",    "1 current  (3 pending)")

    st.markdown("<hr>", unsafe_allow_html=True)

    # Revenue by year
    rev_rows = []
    for y in YEARS:
        col = f"total_revenue_{y}"
        if col in master.columns:
            rev_rows.append({"Year": str(y), "Revenue (JMD)": master[col].sum()})

    def _rev_chart():
        if not rev_rows:
            st.info("No revenue columns found.")
            return
        df_r = pd.DataFrame(rev_rows)
        fig = px.bar(
            df_r, x="Year", y="Revenue (JMD)",
            color_discrete_sequence=["#2563eb"],
            text="Revenue (JMD)",
        )
        fig.update_traces(
            texttemplate="$%{text:,.0f}",
            textposition="outside",
            marker_line_width=0,
        )
        fig.update_xaxes(type="category", tickvals=[str(y) for y in YEARS])
        fig.update_yaxes(tickprefix="$", tickformat=",.0f")
        fig.update_layout(**plotly_defaults(), height=320)
        st.plotly_chart(fig, use_container_width=True)

    chart_container(
        "Annual Revenue (JMD)",
        "<strong>What this shows:</strong> Total revenue generated by your Performance Tool "
        "catalogue each calendar year from 2020 to 2026 in Jamaican Dollars. "
        "Each bar = one full year of sales across all active items. "
        "A rising trend signals healthy category growth; a dip may reflect stock-outs or market shifts.",
        _rev_chart
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        def _cat_chart():
            if cat_col not in master.columns or "lifetime_total_revenue" not in master.columns:
                st.info("Category or revenue data not available.")
                return
            cat_df = (
                master.groupby(cat_col)["lifetime_total_revenue"]
                .sum().sort_values(ascending=True).tail(12).reset_index()
            )
            cat_df.columns = ["Category", "Revenue (JMD)"]
            fig = px.bar(
                cat_df, y="Category", x="Revenue (JMD)", orientation="h",
                color="Revenue (JMD)", color_continuous_scale="Blues",
                labels={"Revenue (JMD)": "Revenue (JMD)"},
            )
            fig.update_layout(**plotly_defaults(), height=380,
                              coloraxis_showscale=False)
            fig.update_xaxes(tickprefix="$", tickformat=",.0f")
            st.plotly_chart(fig, use_container_width=True)

        chart_container(
            "Top Sub-Categories by Lifetime Revenue",
            "<strong>What this shows:</strong> Your 12 highest-revenue sub-categories "
            "ranked by total lifetime sales in JMD. Longer bars = bigger contributors to "
            "your catalogue's revenue. Use this to quickly identify which product types "
            "are your core business and deserve priority in vendor negotiations.",
            _cat_chart
        )

    with col_r:
        def _score_chart():
            if scores is None:
                st.info("Scores not yet available.")
                return
            tier_counts = scores["tier"].value_counts().reset_index()
            tier_counts.columns = ["Tier","Count"]
            color_map = {"High":"#16a34a","Medium":"#d97706","Low":"#dc2626"}
            fig = px.pie(
                tier_counts, names="Tier", values="Count",
                color="Tier", color_discrete_map=color_map,
                hole=0.55,
            )
            fig.update_layout(**plotly_defaults(), height=380, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

        chart_container(
            "Item Viability Tier Breakdown",
            "<strong>What this shows:</strong> How your catalogue items are distributed "
            "across High / Medium / Low viability tiers based on the composite score "
            "(weighted blend of movement, profitability, and convenience). "
            "Adjust the score weights in the sidebar to see how the distribution shifts.",
            _score_chart
        )


# ===========================================================================
# PAGE 2 — SCORING & RANKING
# ===========================================================================
def page_scoring(data, scores):
    st.markdown("## Scoring & Ranking")

    if scores is None:
        st.warning("Could not build scores — check that 02_sage_master.csv exists in processed/.")
        return

    st.markdown(
        "<div class='context-box'><h4>How scores are calculated</h4>"
        "<p>Each item receives three sub-scores (0 to 1) then a weighted composite. "
        "Movement = revenue + volume + transaction frequency over 2022–2024. "
        "Profitability = gross margin % + revenue trend direction. "
        "Convenience = inverse of average lead time (faster = higher score). "
        "Weights are adjustable in the sidebar.</p></div>",
        unsafe_allow_html=True
    )

    # Tier cards
    tier_counts = scores["tier"].value_counts() if "tier" in scores.columns else {}
    c1, c2, c3 = st.columns(3)
    for col, tier, color, desc in [
        (c1, "High",   "#16a34a", "Strong movement + healthy margin + good availability"),
        (c2, "Medium", "#d97706", "Moderate performance — review pricing or reorder frequency"),
        (c3, "Low",    "#dc2626", "Low movement or margin — consider discontinuing or repricing"),
    ]:
        n = tier_counts.get(tier, 0)
        pct = n/len(scores)*100 if len(scores) else 0
        col.markdown(
            f"""<div class='insight-card' style='border-left:4px solid {color};'>
                <div class='label'>{tier} Viability</div>
                <p class='value' style='color:{color};'>{n:,} items</p>
                <p class='sub'>{pct:.1f}% of catalogue</p>
                <p class='sub' style='margin-top:6px;font-size:11px;'>{desc}</p>
            </div>""",
            unsafe_allow_html=True
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])

    with col_l:
        def _top25():
            cat_col = "_cat" if "_cat" in scores.columns else "CATEGORY"
            disp = ["ITEMNO","item_description", cat_col,
                    "composite_score","movement_score",
                    "profitability_score","convenience_score","tier"]
            avail = [c for c in disp if c in scores.columns]
            top = scores.nlargest(25, "composite_score")[avail].copy()
            if cat_col in top.columns:
                top.rename(columns={cat_col: "sub_category"}, inplace=True)
                avail = [c if c != cat_col else "sub_category" for c in avail]

            def _tier_style(v):
                m = {"High":"background:#dcfce7;color:#166534",
                     "Medium":"background:#fef3c7;color:#92400e",
                     "Low":"background:#fee2e2;color:#991b1b"}
                return m.get(v,"")

            styled = (
                top.style
                .format({c: "{:.3f}" for c in
                         ["composite_score","movement_score",
                          "profitability_score","convenience_score"]
                         if c in top.columns})
                .apply(lambda x: x.map(_tier_style), subset=["tier"] if "tier" in top.columns else [])
            )
            st.dataframe(styled, width='stretch', height=440)

        chart_container(
            "Top 25 Items — Composite Score",
            "<strong>What this shows:</strong> Your 25 highest-scoring items across all "
            "three dimensions. Composite score ranges 0–1; items closer to 1 are your "
            "most commercially valuable. The tier column (green/yellow/red) gives an "
            "instant read on priority. Use this list when deciding what to reorder first "
            "or which items to feature in vendor negotiations.",
            _top25
        )

    with col_r:
        def _radar():
            if "composite_score" not in scores.columns:
                return
            cats  = ["Movement","Profitability","Convenience"]
            avg   = scores[["movement_score","profitability_score","convenience_score"]].mean()
            top5m = scores.nlargest(5,"composite_score")[
                        ["movement_score","profitability_score","convenience_score"]
                    ].mean()

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=[avg["movement_score"], avg["profitability_score"], avg["convenience_score"]],
                theta=cats, fill="toself", name="Catalogue Avg",
                line_color="#f4f2ed", fillcolor="rgba(244,242,237,0.15)",
            ))
            fig.add_trace(go.Scatterpolar(
                r=[top5m["movement_score"], top5m["profitability_score"], top5m["convenience_score"]],
                theta=cats, fill="toself", name="Top 5 Avg",
                line_color="#2563eb", fillcolor="rgba(37,99,235,0.18)",
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0,1], gridcolor="#1f2230", linecolor="#d4d0c8"),
                           angularaxis=dict(gridcolor="#1f2230", linecolor="#d4d0c8", tickcolor="#d4d0c8")),
                paper_bgcolor="#111318", plot_bgcolor="#111318", font_family="IBM Plex Sans",
                font_color="#f4f2ed", height=340, margin=dict(l=20,r=20,t=20,b=20),
                legend=dict(orientation="h", y=-0.1, font=dict(color="#d4d0c8")),
            )
            st.plotly_chart(fig, use_container_width=True)

        chart_container(
            "Score Profile — Top 5 vs Catalogue Average",
            "<strong>What this shows:</strong> A radar chart comparing your top 5 items "
            "(blue) to your overall catalogue average (dark). Larger area = stronger "
            "overall score. The gap between the two shapes shows how much better your "
            "best items perform relative to the rest of the catalogue.",
            _radar
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    # Full filterable table
    st.markdown("### Full Scored Catalogue")
    st.markdown(
        "<div class='chart-desc'><strong>How to use this table:</strong> Filter by tier "
        "and set a minimum composite score to narrow down candidates for vendor review. "
        "Sort any column by clicking the header. All scores are 0–1.</div>",
        unsafe_allow_html=True
    )
    fc1, fc2 = st.columns([2, 1])
    tier_filter = fc1.multiselect("Tier", ["High","Medium","Low"],
                                   default=["High","Medium","Low"])
    min_score   = fc2.slider("Min composite score", 0.0, 1.0, 0.0, 0.01)

    filt = scores.copy()
    if "tier" in filt.columns:
        filt = filt[filt["tier"].isin(tier_filter)]
    filt = filt[filt["composite_score"] >= min_score]

    cat_col = "_cat" if "_cat" in filt.columns else "CATEGORY"
    disp = ["ITEMNO","item_description", cat_col,
            "composite_score","movement_score","profitability_score",
            "convenience_score","tier"]
    avail = [c for c in disp if c in filt.columns]
    st.dataframe(
        filt[avail].sort_values("composite_score", ascending=False),
        use_container_width=True, height=380
    )


# ===========================================================================
# PAGE 3 — PREDICTIONS & PO TIMING
# ===========================================================================
def page_predictions(data, preds, scores):
    st.markdown("## Predictions & PO Timing")

    master = data.get("master")
    lead   = data.get("lead")

    st.markdown(
        "<div class='context-box'><h4>What this page answers</h4>"
        "<p>Three questions: (1) What will this item's revenue be in 2026? "
        "(2) When do we need to place the next PO to avoid a stock-out? "
        "(3) If we order today, when does stock arrive?</p></div>",
        unsafe_allow_html=True
    )

    if preds is None or len(preds) == 0:
        st.warning("Need at least 3 years of item history to generate predictions.")
        return

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Items Forecast",        f"{len(preds):,}")
    c2.metric("Total Predicted Rev 2026", f"${preds['rev_pred_2026'].sum():,.0f} JMD")
    growing   = (preds["trend"] == "Growing").sum()
    declining = (preds["trend"] == "Declining").sum()
    c3.metric("Growing Items",    f"{growing:,}  ({growing/len(preds)*100:.0f}%)")
    c4.metric("Declining Items",  f"{declining:,}  ({declining/len(preds)*100:.0f}%)")

    st.markdown("<hr>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        def _trend_donut():
            td = preds["trend"].value_counts().reset_index()
            td.columns = ["Trend","Count"]
            fig = px.pie(
                td, names="Trend", values="Count", hole=0.55,
                color="Trend",
                color_discrete_map={"Growing":"#16a34a","Stable":"#2563eb","Declining":"#dc2626"},
            )
            fig.update_layout(**plotly_defaults(), height=280, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

        chart_container(
            "2026 Demand Trend — Catalogue Split",
            "<strong>What this shows:</strong> Of all items with enough history to forecast, "
            "this donut shows how many are on a growing, stable, or declining revenue "
            "trajectory heading into 2026. Green = prioritise for reorder. "
            "Red = review before placing large POs.",
            _trend_donut
        )

    with col_r:
        def _profit_scatter():
            df_p = preds[(preds["pt_cost"] > 0) & (preds["current_sell_price"] > 0)].copy()
            if df_p.empty:
                st.info("Need PT cost and selling price data for this chart.")
                return
            df_p["pred_margin_jmd"] = (
                (df_p["current_sell_price"] - df_p["pt_cost"]) * df_p["rev_pred_2026"]
                / (df_p["current_sell_price"] + 0.01)
            )
            fig = px.scatter(
                df_p.head(200),
                x="rev_pred_2026", y="pred_margin_jmd",
                color="trend",
                color_discrete_map={"Growing":"#16a34a","Stable":"#2563eb","Declining":"#dc2626"},
                hover_name="item_description",
                labels={"rev_pred_2026":"Predicted 2026 Revenue (JMD)",
                        "pred_margin_jmd":"Predicted 2026 Margin (JMD)",
                        "trend":"Trend"},
            )
            fig.update_layout(**plotly_defaults(), height=280)
            fig.update_xaxes(tickprefix="$", tickformat=",.0f")
            fig.update_yaxes(tickprefix="$", tickformat=",.0f")
            st.plotly_chart(fig, use_container_width=True)

        chart_container(
            "Predicted 2026 Revenue vs Predicted Margin",
            "<strong>What this shows:</strong> Each dot is one item. X-axis = how much "
            "revenue it's predicted to generate in 2026. Y-axis = estimated gross margin "
            "on that revenue. Best items sit top-right (high revenue + high margin). "
            "Color shows trend direction.",
            _profit_scatter
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    # PO TIMING SECTION
    st.markdown("### PO Timing — When to Order & When It Arrives")
    st.markdown(
        "<div class='chart-desc'><strong>What this shows:</strong> For each item, "
        "using its average lead time from your PO history, the table calculates: "
        "(a) the estimated ETA if you ordered <em>today</em>, and "
        "(b) a suggested reorder trigger based on predicted 2026 monthly demand. "
        "If no lead time exists in PO data, the default of "
        f"{DEFAULT_PT_LEAD_DAYS} days is used.</div>",
        unsafe_allow_html=True
    )

    # Merge preds with lead time from PO history
    df_po = preds.copy()
    if lead is not None and "avg_lead_time_days" in lead.columns:
        df_po = df_po.merge(lead[["ITEMNO","avg_lead_time_days","po_order_count"]],
                            on="ITEMNO", how="left")
    else:
        df_po["avg_lead_time_days"] = np.nan
        df_po["po_order_count"]     = 0

    # Use vendor country defaults when item lead time is missing
    combined = data.get("combined")
    if combined is not None and "country" in combined.columns:
        vendor_country = combined[["item_number","country"]].drop_duplicates(subset=["item_number"], keep="first")
        df_po = df_po.merge(vendor_country, left_on="ITEMNO", right_on="item_number", how="left")
    else:
        df_po["country"] = None

    df_po["avg_lead_time_days"] = df_po["avg_lead_time_days"].fillna(
        df_po["country"].fillna("")
              .str.contains("germany", case=False, na=False)
              .replace({True: 42, False: np.nan})
    )
    df_po["avg_lead_time_days"] = df_po["avg_lead_time_days"].fillna(
        df_po["country"].fillna("")
              .str.contains("iran|china|chinese", case=False, na=False)
              .replace({True: 90, False: np.nan})
    )
    df_po["avg_lead_time_days"] = df_po["avg_lead_time_days"].fillna(DEFAULT_PT_LEAD_DAYS)

    df_po["eta_if_order_today"] = df_po["avg_lead_time_days"].apply(
        lambda x: (datetime.today() + timedelta(days=int(x))).strftime("%b %d, %Y")
    )

    # Monthly demand estimate from 2026 prediction
    df_po["monthly_demand_2026"] = (df_po["rev_pred_2026"] / (df_po["current_sell_price"].replace(0, np.nan)) / 12).round(1)

    # Reorder point = lead time (days) × daily demand
    df_po["daily_demand"] = (df_po["monthly_demand_2026"] / 30).round(2)
    df_po["reorder_point_units"] = (df_po["daily_demand"] * df_po["avg_lead_time_days"]).round(0)

    show_cols = ["ITEMNO","item_description","trend",
                 "rev_pred_2026","current_sell_price",
                 "country","avg_lead_time_days","eta_if_order_today",
                 "monthly_demand_2026","reorder_point_units","po_order_count"]
    avail_cols = [c for c in show_cols if c in df_po.columns]

    # Filter controls
    pc1, pc2 = st.columns([2,1])
    trend_f = pc1.multiselect("Filter by trend", ["Growing","Stable","Declining"],
                               default=["Growing","Stable","Declining"])
    min_rev  = pc2.number_input("Min predicted 2026 revenue (JMD)", 0, value=0, step=1000)

    df_filtered = df_po[
        df_po["trend"].isin(trend_f) & (df_po["rev_pred_2026"] >= min_rev)
    ]

    fmt = {
        "rev_pred_2026":       "${:,.0f}",
        "current_sell_price":  "${:,.2f}",
        "avg_lead_time_days":  "{:.0f} days",
        "monthly_demand_2026": "{:.1f} units",
        "reorder_point_units": "{:.0f} units",
    }
    st.dataframe(
        df_filtered[avail_cols]
        .sort_values("rev_pred_2026", ascending=False)
        .style.format({k: v for k, v in fmt.items() if k in avail_cols}),
        use_container_width=True, height=420
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # Top 30 2026 forecast bar chart
    def _top30_chart():
        top = preds.nlargest(30, "rev_pred_2026").copy()
        top["label"] = top["ITEMNO"] + " " + top["item_description"].str[:25]
        fig = px.bar(
            top.sort_values("rev_pred_2026"),
            y="label", x="rev_pred_2026",
            orientation="h",
            color="trend",
            color_discrete_map={"Growing":"#16a34a","Stable":"#2563eb","Declining":"#dc2626"},
            labels={"rev_pred_2026":"Predicted 2026 Revenue (JMD)","label":""},
        )
        fig.update_layout(**plotly_defaults(), height=560)
        fig.update_xaxes(tickprefix="$", tickformat=",.0f")
        st.plotly_chart(fig, use_container_width=True)

    chart_container(
        "Top 30 Items — 2026 Revenue Forecast",
        "<strong>What this shows:</strong> The 30 items predicted to generate the most "
        "revenue in 2026, coloured by their trend direction. These are your highest-priority "
        "items for PO planning. Growing items (green) in this list should be ordered with "
        "additional buffer stock — demand is rising.",
        _top30_chart
    )


# ===========================================================================
# PAGE 4 — PRICE ANALYSIS
# ===========================================================================
def page_price_analysis(data, ronix_ff):
    st.markdown("## Price Analysis")
    st.markdown(
        "<div class='context-box'><h4>What this page covers</h4>"
        "<p>Three lenses on pricing: how your average selling price has changed year "
        "over year, how your sell price compares to vendor cost, and where your "
        "margins sit across the catalogue. All prices in JMD unless noted.</p></div>",
        unsafe_allow_html=True
    )

    master = data.get("master")
    pt_cat = data.get("pt_cat")
    ronix  = data.get("ronix")

    tab1, tab2, tab3 = st.tabs([
        "📅  Year-over-Year Price Trend",
        "🔄  Vendor Price Comparison",
        "📐  Margin Analysis",
    ])

    # ---- TAB 1: YoY ----
    with tab1:
        st.markdown("")
        st.markdown(
            "<div class='chart-desc'><strong>What this shows:</strong> The average "
            "selling price across your catalogue each year from 2020 to 2026 in JMD, "
            "with a line showing the percentage change from the prior year. "
            "A rising bar with a positive % line means you've been able to grow "
            "prices — a healthy sign. A dip in the % line can flag competitive "
            "pressure or mix shifts toward lower-priced items.</div>",
            unsafe_allow_html=True
        )
        if master is not None:
            avg_rows = []
            for y in YEARS:
                col = f"avg_price_{y}"
                if col in master.columns:
                    vals = master[col][master[col] > 0]
                    if len(vals):
                        avg_rows.append({"Year": str(y), "Avg Selling Price (JMD)": vals.mean()})

            if avg_rows:
                df_yoy = pd.DataFrame(avg_rows)
                df_yoy["YoY Change %"] = df_yoy["Avg Selling Price (JMD)"].pct_change() * 100

                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Bar(
                    x=df_yoy["Year"], y=df_yoy["Avg Selling Price (JMD)"],
                    name="Avg Selling Price", marker_color="#2563eb",
                ), secondary_y=False)
                fig.add_trace(go.Scatter(
                    x=df_yoy["Year"], y=df_yoy["YoY Change %"],
                    name="YoY Change %",
                    line=dict(color="#e87c55", width=2.5),
                    mode="lines+markers",
                ), secondary_y=True)
                fig.update_xaxes(type="category",
                                 tickvals=[str(y) for y in YEARS],
                                 ticktext=[str(y) for y in YEARS])
                fig.update_yaxes(title_text="Avg Price (JMD)", tickprefix="$",
                                 secondary_y=False)
                fig.update_yaxes(title_text="YoY %", ticksuffix="%",
                                 secondary_y=True)
                layout = plotly_defaults()
                layout["legend"] = {**layout.get("legend", {}),
                                     "orientation": "h", "y": 1.08}
                fig.update_layout(**layout, height=380)
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("**Year-by-Year Summary Table**")
                st.dataframe(
                    df_yoy.style.format({
                        "Avg Selling Price (JMD)": "${:,.2f}",
                        "YoY Change %": "{:+.2f}%",
                    }),
                    use_container_width=True
                )

    # ---- TAB 2: Vendor Price Comparison ----
    with tab2:
        st.markdown("")
        st.markdown(
            "<div class='chart-desc'><strong>What this shows:</strong> A scatter plot of "
            "your average selling price (Y-axis) vs the Performance Tool CPL35 cost price "
            "(X-axis) for each matched item. The dashed diagonal line = break-even (sell "
            "price equals cost). Dots <em>above</em> the line = positive margin. "
            "Dots <em>below</em> = you're selling at a loss relative to PT cost — "
            "those items need immediate attention. Colour intensity shows margin %.</div>",
            unsafe_allow_html=True
        )

        if master is not None and pt_cat is not None:
            if "item_number" in pt_cat.columns and "ITEMNO" in master.columns:
                price_col = "pt_cpl35_price" if "pt_cpl35_price" in pt_cat.columns else (
                    "pt_current_price" if "pt_current_price" in pt_cat.columns else None
                )
                if price_col is not None:
                    pc = pt_cat[["item_number", price_col, "item_description"]].dropna(subset=[price_col]).copy()
                    pc = pc.rename(columns={price_col: "pt_cpl35_price"})
                    merged = master.merge(pc, left_on="ITEMNO", right_on="item_number", how="inner")
                    merged = merged[(merged["lifetime_avg_price"] > 0) & (merged["pt_cpl35_price"] > 0)].copy()
                else:
                    merged = pd.DataFrame()

                if len(merged):
                    # Note: lifetime_avg_price is JMD; pt_cpl35_price is USD.
                    # Load exchange rate if available for fair comparison.
                    _xrate = None
                    _xrate_file = DATA_DIR.parent / "Search BOJ Counter Rates.csv"
                    if _xrate_file.exists():
                        try:
                            _xdf = pd.read_csv(_xrate_file)
                            _xdf["Date"] = pd.to_datetime(_xdf["Date"], format="%d %b %Y", errors="coerce")
                            _xdf = _xdf.dropna(subset=["Date"]).sort_values("Date")
                            _xrate = float(_xdf.iloc[-1]["DMD & TT"])
                        except Exception:
                            _xrate = None

                    if _xrate and _xrate > 0:
                        merged["pt_cpl35_price_jmd"] = merged["pt_cpl35_price"] * _xrate
                        merged["margin_pct"] = (
                            (merged["lifetime_avg_price"] - merged["pt_cpl35_price_jmd"])
                            / merged["lifetime_avg_price"] * 100
                        ).round(1)
                        cost_col_display = "pt_cpl35_price_jmd"
                        cost_label = f"Avg PT Cost (JMD @ {_xrate:.2f})"
                    else:
                        merged["margin_pct"] = (
                            (merged["lifetime_avg_price"] - merged["pt_cpl35_price"])
                            / merged["lifetime_avg_price"] * 100
                        ).round(1)
                        cost_col_display = "pt_cpl35_price"
                        cost_label = "Avg PT CPL35 Cost (USD — exchange rate file not found)"

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Matched Items",    f"{len(merged):,}")
                    c2.metric("Avg Sell Price (JMD)", f"${merged['lifetime_avg_price'].mean():,.2f}")
                    c3.metric(cost_label,         f"${merged[cost_col_display].mean():,.2f}")
                    c4.metric("Items Below Cost", f"{(merged['margin_pct'] < 0).sum():,}")

                    fig = px.scatter(
                        merged.head(300),
                        x="pt_cpl35_price", y="lifetime_avg_price",
                        color="margin_pct",
                        color_continuous_scale="RdYlGn",
                        range_color=[-20, 60],
                        hover_name="item_description_x",
                        hover_data={"margin_pct": ":.1f%"},
                        labels={
                            "pt_cpl35_price":     "PT CPL35 Cost",
                            "lifetime_avg_price": "Your Avg Selling Price",
                            "margin_pct":         "Margin %",
                        },
                    )
                    max_v = max(merged["pt_cpl35_price"].max(), merged["lifetime_avg_price"].max())
                    fig.add_shape(type="line", x0=0,y0=0, x1=max_v,y1=max_v,
                                  line=dict(color="#6b6860",dash="dash",width=1))
                    fig.update_layout(**plotly_defaults(), height=420,
                                       coloraxis_colorbar_title="Margin %")
                    st.plotly_chart(fig, use_container_width=True)

        # Ronix FOB comparison
        if ronix is not None and "fob_unit_price_usd" in ronix.columns:
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown(
                "<div class='chart-desc'><strong>Ronix FOB vs Estimated Landed:</strong> "
                "Ronix prices are FOB (Free On Board) from Iran — the price does NOT include "
                "freight, insurance, or import duties to your market. The 'Estimated Landed' "
                f"column applies your current freight factor ({ronix_ff}×) set in the sidebar. "
                "Adjust the slider to model different freight scenarios.</div>",
                unsafe_allow_html=True
            )
            r = ronix[["item_number","item_description","fob_unit_price_usd"]].dropna().copy()
            r["estimated_landed_usd"] = (r["fob_unit_price_usd"] * ronix_ff).round(2)
            r["freight_uplift_usd"]   = (r["estimated_landed_usd"] - r["fob_unit_price_usd"]).round(2)
            st.dataframe(
                r.style.format({
                    "fob_unit_price_usd":    "${:.2f}",
                    "estimated_landed_usd":  "${:.2f}",
                    "freight_uplift_usd":    "+${:.2f}",
                }),
                use_container_width=True, height=340
            )

    # ---- TAB 3: Margin Analysis ----
    with tab3:
        st.markdown("")
        st.markdown(
            "<div class='chart-desc'><strong>What this shows:</strong> Gross margin "
            "distribution across your catalogue — how much profit (as a %) sits "
            "between your cost (PT CPL35 price) and your average selling price. "
            "The red dashed line = catalogue average margin. Items to the left of that "
            "line are below-average margin contributors. The bar chart breaks this down "
            "by sub-category so you can see which product types drive the most profit.</div>",
            unsafe_allow_html=True
        )

        if master is not None and "performance_tool_price" in master.columns:
            cat_col = "Sub Category Name" if "Sub Category Name" in master.columns else "CATEGORY"

            # Convert PT cost to JMD for fair margin calculation
            # Gross Margin % = [(Revenue - COGS) / Revenue] × 100
            # Revenue proxy = lifetime_avg_price (JMD sell price)
            # COGS proxy    = performance_tool_price × exchange rate (USD → JMD)
            _xrate_m = None
            _xrate_file_m = DATA_DIR.parent / "Search BOJ Counter Rates.csv"
            if _xrate_file_m.exists():
                try:
                    _xdf_m = pd.read_csv(_xrate_file_m)
                    _xdf_m["Date"] = pd.to_datetime(_xdf_m["Date"], format="%d %b %Y", errors="coerce")
                    _xdf_m = _xdf_m.dropna(subset=["Date"]).sort_values("Date")
                    _xrate_m = float(_xdf_m.iloc[-1]["DMD & TT"])
                except Exception:
                    _xrate_m = None

            df_m = master[
                (master["lifetime_avg_price"] > 0) & (master["performance_tool_price"] > 0)
            ].copy()

            if _xrate_m and _xrate_m > 0:
                df_m["_cogs_jmd"] = df_m["performance_tool_price"] * _xrate_m
                xrate_note = f"PT cost converted to JMD at {_xrate_m:.2f}"
            else:
                df_m["_cogs_jmd"] = df_m["performance_tool_price"]
                xrate_note = "Warning: exchange rate file not found — using PT cost as-is"

            # Gross Margin % = [(Revenue − COGS) / Revenue] × 100
            df_m["gross_margin_pct"] = (
                (df_m["lifetime_avg_price"] - df_m["_cogs_jmd"])
                / df_m["lifetime_avg_price"] * 100
            ).clip(-200, 100).round(2)

            st.caption(f"Margin formula: [(Sell Price − PT Cost) / Sell Price] × 100. {xrate_note}.")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Avg Gross Margin",    f"{df_m['gross_margin_pct'].mean():.1f}%")
            c2.metric("Median Margin",       f"{df_m['gross_margin_pct'].median():.1f}%")
            c3.metric("Items > 40% Margin",  f"{(df_m['gross_margin_pct'] > 40).sum():,}")
            c4.metric("Items < 10% Margin",  f"{(df_m['gross_margin_pct'] < 10).sum():,}")

            fig = px.histogram(
                df_m, x="gross_margin_pct", nbins=40,
                color_discrete_sequence=["#2563eb"],
                labels={"gross_margin_pct": "Gross Margin %"},
            )
            avg_m = df_m["gross_margin_pct"].mean()
            fig.add_vline(x=avg_m, line_dash="dash", line_color="#dc2626",
                          annotation_text=f"Avg {avg_m:.1f}%",
                          annotation_position="top right")
            fig.update_layout(**plotly_defaults(), height=300)
            st.plotly_chart(fig, use_container_width=True)

            if cat_col in df_m.columns:
                cat_m = (
                    df_m.groupby(cat_col)["gross_margin_pct"]
                    .mean().sort_values().reset_index()
                )
                cat_m.columns = ["Sub Category","Avg Margin %"]
                fig2 = px.bar(
                    cat_m, y="Sub Category", x="Avg Margin %",
                    orientation="h",
                    color="Avg Margin %",
                    color_continuous_scale="RdYlGn",
                    range_color=[0, 60],
                )
                fig2.update_layout(**plotly_defaults(), height=max(300, len(cat_m)*22),
                                    coloraxis_showscale=False)
                fig2.update_xaxes(ticksuffix="%")
                st.plotly_chart(fig2, use_container_width=True)


# ===========================================================================
# PAGE 5 — ITEM DEEP DIVE
# ===========================================================================
def page_item_deep_dive(data, scores, preds):
    st.markdown("## Item Deep Dive")
    st.markdown(
        "<div class='context-box'><h4>Full profile for a single item</h4>"
        "<p>Select any item to see its complete sales history, scores, margin, "
        "current selling price, 2026 revenue forecast, and PO timing estimate. "
        "Current selling price uses 2026 data; falls back to 2025 if 2026 is empty.</p>"
        "</div>",
        unsafe_allow_html=True
    )

    master = data.get("master")
    lead   = data.get("lead")
    if master is None:
        st.warning("No master data loaded.")
        return

    # Single selector — large with description
    items = master[["ITEMNO","item_description"]].dropna().copy()
    items["label"] = items["ITEMNO"] + "  —  " + items["item_description"].str[:60]
    selected_label = st.selectbox("Search or select an item", items["label"].tolist())
    item_no = selected_label.split("  —  ")[0].strip()

    rows = master[master["ITEMNO"] == item_no]
    if rows.empty:
        st.warning("Item not found.")
        return
    row = rows.iloc[0]

    # Current selling price: 2026 > 2025 > lifetime
    price_2026 = pd.to_numeric(row.get("avg_price_2026", 0), errors="coerce") or 0
    price_2025 = pd.to_numeric(row.get("avg_price_2025", 0), errors="coerce") or 0
    current_price = price_2026 if price_2026 > 0 else (price_2025 if price_2025 > 0 else
                    pd.to_numeric(row.get("lifetime_avg_price", 0), errors="coerce") or 0)
    price_source  = "2026" if price_2026 > 0 else ("2025" if price_2025 > 0 else "Lifetime Avg")

    pt_cost = pd.to_numeric(row.get("performance_tool_price", 0), errors="coerce") or 0
    margin  = ((current_price - pt_cost) / current_price * 100) if current_price > 0 else 0

    # Lead time for this item
    item_lt = DEFAULT_PT_LEAD_DAYS
    if lead is not None and "avg_lead_time_days" in lead.columns:
        lt_row = lead[lead["ITEMNO"] == item_no]
        if not lt_row.empty:
            item_lt = lt_row["avg_lead_time_days"].iloc[0]
    eta_date, eta_days = get_eta(item_lt)

    # Header card
    cat_col = "Sub Category Name" if "Sub Category Name" in master.columns else "CATEGORY"
    st.markdown(
        f"""<div class='insight-card' style='border-left:4px solid #2563eb;'>
            <div class='label'>Item {item_no}</div>
            <p class='value'>{row.get('item_description', 'N/A')}</p>
            <p class='sub'>
                Sub-category: <strong>{row.get(cat_col,'N/A')}</strong> &nbsp;|&nbsp;
                Unit: <strong>{row.get('STOCKUNIT','N/A')}</strong> &nbsp;|&nbsp;
                PT CPL35 Cost: <strong>${pt_cost:,.2f}</strong> &nbsp;|&nbsp;
                Current Sell Price ({price_source}): <strong>${current_price:,.2f}</strong> &nbsp;|&nbsp;
                Gross Margin: <strong>{margin:.1f}%</strong>
            </p>
        </div>""",
        unsafe_allow_html=True
    )

    # Score + ETA metrics row
    s_row = scores[scores["ITEMNO"] == item_no] if scores is not None else pd.DataFrame()
    c1, c2, c3, c4, c5 = st.columns(5)
    if not s_row.empty:
        s = s_row.iloc[0]
        c1.metric("Composite Score",    f"{s.get('composite_score',0):.3f}")
        c2.metric("Movement Score",     f"{s.get('movement_score',0):.3f}")
        c3.metric("Profitability Score",f"{s.get('profitability_score',0):.3f}")
        c4.metric("Convenience Score",  f"{s.get('convenience_score',0):.3f}")
    c5.metric(f"ETA if ordered today", f"{eta_date}  ({eta_days}d)")

    st.markdown("<hr>", unsafe_allow_html=True)

    # Revenue + Volume charts side by side
    rev_data = [(str(y), pd.to_numeric(row.get(f"total_revenue_{y}", 0), errors="coerce") or 0)
                for y in YEARS if f"total_revenue_{y}" in row.index]
    vol_data = [(str(y), pd.to_numeric(row.get(f"total_volume_{y}", 0), errors="coerce") or 0)
                for y in YEARS if f"total_volume_{y}" in row.index]
    frq_data = [(str(y), pd.to_numeric(row.get(f"sales_frequency_{y}", 0), errors="coerce") or 0)
                for y in YEARS if f"sales_frequency_{y}" in row.index]

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("### Revenue History (JMD)")
        st.markdown(
            "<div class='chart-desc'><strong>What this shows:</strong> Year-by-year "
            "revenue for this item from 2020 to 2026. The blue bar (if shown) is the "
            "model's 2026 prediction. A consistent upward trend = strong candidate "
            "for increased stock levels.</div>",
            unsafe_allow_html=True
        )
        if rev_data:
            df_rev = pd.DataFrame(rev_data, columns=["Year","Revenue"])
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_rev["Year"], y=df_rev["Revenue"],
                name="Actual", marker_color="#2563eb",
            ))
            # Add 2026 prediction if available
            if preds is not None:
                p_row = preds[preds["ITEMNO"] == item_no]
                if not p_row.empty:
                    pred_2026 = p_row["rev_pred_2026"].iloc[0]
                    fig.add_trace(go.Bar(
                        x=["2026"], y=[pred_2026],
                        name="Predicted 2026", marker_color="#93c5fd",
                    ))
                    fig.add_annotation(
                        x="2026", y=pred_2026,
                        text=f"Pred: ${pred_2026:,.0f}",
                        showarrow=False, yshift=10,
                        font=dict(size=11, color="#2563eb")
                    )
            fig.update_xaxes(type="category", tickvals=[str(y) for y in YEARS])
            fig.update_yaxes(tickprefix="$", tickformat=",.0f")
            fig.update_layout(**plotly_defaults(), height=280, barmode="overlay")
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("### Units Sold & Transaction Frequency")
        st.markdown(
            "<div class='chart-desc'><strong>What this shows:</strong> Bars = units sold "
            "per year. Line = number of separate transactions (orders). "
            "High bars with low line = a few large bulk orders. "
            "High line = frequent small orders — good sign for steady demand.</div>",
            unsafe_allow_html=True
        )
        if vol_data and frq_data:
            df_v = pd.DataFrame(vol_data, columns=["Year","Volume"])
            df_f = pd.DataFrame(frq_data, columns=["Year","Freq"])
            df_vf = df_v.merge(df_f, on="Year")
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=df_vf["Year"], y=df_vf["Volume"],
                                  name="Units Sold", marker_color="#2563eb"),
                          secondary_y=False)
            fig.add_trace(go.Scatter(x=df_vf["Year"], y=df_vf["Freq"],
                                      name="Transactions",
                                      line=dict(color="#e87c55", width=2),
                                      mode="lines+markers"),
                          secondary_y=True)
            fig.update_xaxes(type="category", tickvals=[str(y) for y in YEARS])
            fig.update_yaxes(title_text="Units", secondary_y=False)
            fig.update_yaxes(title_text="Transactions", secondary_y=True)
            fig.update_layout(**plotly_defaults(), height=280)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Avg selling price trend
    st.markdown("### Average Selling Price Trend (JMD)")
    st.markdown(
        "<div class='chart-desc'><strong>What this shows:</strong> How the average "
        "unit selling price for this item has changed each year. The red dashed line "
        "is the current PT CPL35 cost — any year where the green line dips below the "
        "red line means you sold this item at a loss that year.</div>",
        unsafe_allow_html=True
    )
    ap_data = [(str(y), pd.to_numeric(row.get(f"avg_price_{y}", 0), errors="coerce") or 0)
               for y in YEARS if f"avg_price_{y}" in row.index]
    if ap_data:
        df_ap = pd.DataFrame(ap_data, columns=["Year","Avg Price"])
        df_ap = df_ap[df_ap["Avg Price"] > 0]
        fig = px.line(df_ap, x="Year", y="Avg Price",
                      markers=True, color_discrete_sequence=["#16a34a"])
        if pt_cost > 0:
            fig.add_hline(y=pt_cost, line_dash="dash", line_color="#dc2626",
                          annotation_text=f"PT Cost ${pt_cost:.2f}",
                          annotation_position="bottom right")
        fig.update_xaxes(type="category")
        fig.update_yaxes(tickprefix="$")
        fig.update_layout(**plotly_defaults(), height=240)
        st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# PAGE 6 — VENDOR COMPARISON
# ===========================================================================
def page_vendor_comparison(data, scores):
    st.markdown("## Vendor Comparison")
    st.markdown(
        "<div class='context-box'><h4>Side-by-side vendor assessment</h4>"
        "<p>Compares Performance Tools (current), CPL35 (new), Metabo, and Ronix across price, "
        "lead time, catalogue depth, and risk flags. CPL data comes from the CPL35 Jan 02 price list, "
        "while Performance Tools data comes from the Performance_Tools master file.</p></div>",
        unsafe_allow_html=True
    )

    master = data.get("master")
    cpl    = data.get("pt_cat")
    po     = data.get("po")
    metabo = data.get("metabo")
    ronix  = data.get("ronix")
    lead   = data.get("lead")



    # Vendor profile cards
    avg_lt = None
    if lead is not None and "avg_lead_time_days" in lead.columns:
        avg_lt = lead["avg_lead_time_days"].mean()
    elif po is not None and "lead_time_days" in po.columns:
        avg_lt = po["lead_time_days"].mean()

    st.markdown("### Vendor Profiles at a Glance")
    st.markdown(
        "<div class='chart-desc'><strong>What this shows:</strong> Key facts about each "
        "vendor — where they're based, how many items they carry, pricing type, and "
        "any risk flags (expired prices, phase-out items). This is the executive summary "
        "row your director will want to see first.</div>",
        unsafe_allow_html=True
    )

    def count_true_flags(series):
        if series is None:
            return 0
        if pd.api.types.is_bool_dtype(series):
            return int(series.sum())
        normalized = series.astype(str).str.lower().str.strip()
        return int(normalized.isin({"true", "1", "yes", "y", "t"}).sum())

    cpl_count = len(cpl) if cpl is not None else 0
    cpl_flags = "TBD"
    if cpl is not None:
        cpl_exp = count_true_flags(cpl["price_expired"]) if "price_expired" in cpl.columns else 0
        cpl_po  = count_true_flags(cpl["phase_out_flag"]) if "phase_out_flag" in cpl.columns else 0
        cpl_flags = f"{cpl_exp} expired prices · {cpl_po} phase-out items" if (cpl_exp + cpl_po) else "No immediate flags"

    profiles = []
    profiles.append({
        "Vendor":         "CPL",
        "Country":        "USA",
        "Status":         "New",
        "Catalogue Size": f"{cpl_count:,} items" if cpl_count else "Price list not yet provided",
        "Avg Lead Time":  "TBD",
        "Price Basis":    "CPL35 Jan 02 price list" if cpl_count else "TBD",
        "Risk Flags":     cpl_flags,
    })
    if metabo is not None:
        exp  = count_true_flags(metabo["price_expired"]) if "price_expired" in metabo.columns else 0
        po_f = count_true_flags(metabo["phase_out_flag"]) if "phase_out_flag" in metabo.columns else 0
        profiles.append({
            "Vendor":         "Metabo",
            "Country":        "Germany",
            "Status":         "New",
            "Catalogue Size": f"{len(metabo):,} items",
            "Avg Lead Time":  "~45–60 days (intl)",
            "Price Basis":    "Landed (distributor)",
            "Risk Flags":     f"{exp} expired prices · {po_f} phase-out items" if (exp+po_f) else "None",
        })
    if ronix is not None:
        profiles.append({
            "Vendor":         "Ronix",
            "Country":        "Iran",
            "Status":         "New",
            "Catalogue Size": f"{len(ronix):,} items",
            "Avg Lead Time":  "~60–90 days (FOB)",
            "Price Basis":    "FOB — add freight + duty",
            "Risk Flags":     "Sanctions risk · currency volatility · longer transit",
        })

    if master is not None:
        profiles.append({
            "Vendor":         "Performance Tools",
            "Country":        "USA",
            "Status":         "Current",
            "Catalogue Size": f"{len(master):,} items",
            "Avg Lead Time":  f"{avg_lt:.0f} days" if avg_lt else f"~{DEFAULT_PT_LEAD_DAYS}d default",
            "Price Basis":    "External cost data from various sources, updated periodically",
            "Risk Flags":     "None",
        })


    # --- Styled vendor profile table (white text, readable on dark bg) ---
    df_prof = pd.DataFrame(profiles)
    col_widths = {"Vendor": "120px", "Country": "80px", "Status": "80px",
                  "Catalogue Size": "120px", "Avg Lead Time": "120px",
                  "Price Basis": "220px", "Risk Flags": "260px"}
    header_cells = "".join(
        f"<th style='padding:10px 14px;text-align:left;border-bottom:2px solid #2563eb;"
        f"font-family:IBM Plex Mono,monospace;font-size:12px;letter-spacing:0.08em;"
        f"text-transform:uppercase;color:#93c5fd;min-width:{col_widths.get(c,"100px")}'>"
        f"{c}</th>"
        for c in df_prof.columns
    )
    status_colors = {"Current": "#16a34a", "New": "#f59e0b"}
    rows_html = ""
    for i, row in df_prof.iterrows():
        cells = ""
        for col in df_prof.columns:
            val = str(row[col])
            extra = ""
            if col == "Status":
                color = status_colors.get(val, "#d4d0c8")
                extra = f"color:{color};font-weight:600;"
            elif col == "Vendor":
                extra = "font-weight:700;font-size:16px;"
            elif col == "Risk Flags" and any(w in val for w in ["Sanctions","expired","phase-out"]):
                extra = "color:#fbbf24;"
            cells += (
                f"<td style='padding:10px 14px;border-bottom:1px solid #1f2230;"
                f"font-size:15px;color:#f4f2ed;{extra}'>{val}</td>"
            )
        rows_html += f"<tr>{cells}</tr>"

    st.markdown(
        f"<div style='overflow-x:auto;'>"
        f"<table style='width:100%;border-collapse:collapse;background:#1a1814;"
        f"border-radius:10px;overflow:hidden;font-family:IBM Plex Sans,sans-serif;'>"
        f"<thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        f"</table></div>",
        unsafe_allow_html=True
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("### Cost Comparison — Performance Tools vs Matched Vendors")
    st.markdown(
        "<div class='chart-desc'><strong>What this shows:</strong> Item-level comparison across vendors. "
        "Two-pass item-number match: <b>Pass 1</b> matches trimmed Sage ITEMNO against vendor item_number (keep ≥80%). "
        "<b>Pass 2</b> uses the Sage vendor item number from PO history for any items that scored below 80% in pass 1. "
        "A grouped bar chart shows PT cost vs each matched vendor side-by-side.</div>",
        unsafe_allow_html=True
    )

    # -----------------------------------------------------------------------
    # Two-pass fuzzy item-number match
    # Pass 1: Sage ITEMNO (trimmed) vs vendor item_number  -> keep >= 80%
    # Pass 2: Sage vendor_item_number (from PO sheet)      -> keep >= 80%
    # -----------------------------------------------------------------------
    FUZZY_HIGH = 80

    def _token_sort_score(a: str, b: str) -> float:
        """SequenceMatcher token-sort ratio 0-100."""
        ta = " ".join(sorted(str(a).strip().lower().split()))
        tb = " ".join(sorted(str(b).strip().lower().split()))
        return SequenceMatcher(None, ta, tb).ratio() * 100

    @st.cache_data(show_spinner=False)
    def _build_vendor_index(_df, price_col):
        """Pre-build (item_number_clean, description_clean, price) tuples."""
        if _df is None or _df.empty:
            return []
        rows = []
        for _, r in _df.iterrows():
            ino  = str(r.get("item_number", "")).strip().lower()
            desc = str(r.get("item_description", "")).strip().lower()
            p    = pd.to_numeric(r.get(price_col, None), errors="coerce")
            rows.append((ino, desc, float(p) if pd.notna(p) and p > 0 else None, r))
        return rows

    def best_itemno_match(query: str, vendor_df: pd.DataFrame):
        """Match query against vendor item_number only (for item-number pass)."""
        if vendor_df is None or vendor_df.empty or not query.strip():
            return None, 0
        best_row, best_score = None, 0.0
        q = query.strip().lower()
        for _, vrow in vendor_df.iterrows():
            s = _token_sort_score(q, str(vrow.get("item_number", "")).strip().lower())
            if s > best_score:
                best_score, best_row = s, vrow
        return best_row, best_score

    def best_desc_match(query_desc: str, vendor_index: list, threshold: float = 55.0):
        """Match PT item description against vendor item descriptions.
        Returns (price, score) of the best description match above threshold."""
        if not query_desc.strip() or not vendor_index:
            return None, 0
        best_price, best_score = None, 0.0
        q = query_desc.strip().lower()
        for ino, desc, price, _ in vendor_index:
            if not desc:
                continue
            s = _token_sort_score(q, desc)
            if s > best_score:
                best_score, best_price = s, price
        if best_score >= threshold:
            return best_price, round(best_score, 1)
        return None, 0

    # Build comparison data
    if master is not None:
        po = data.get("po")

        # Build sage_vendor_itemno lookup
        # Primary: vendor_item_number is now baked into 02_sage_master.csv
        #          (mapped + text-cleaned by load_vendor_data.py at ingest time)
        # Fallback: PO transactions sheet for any items not in master
        import re as _re
        _vin_pat = _re.compile(r"^([A-Za-z0-9][A-Za-z0-9\-\._]*)")

        sage_vendor_ino = {}

        # Source 1 — master CSV (fast, already cleaned)
        if "vendor_item_number" in master.columns:
            for _, _mr in master[master["ITEMNO"].notna()].iterrows():
                k = str(_mr["ITEMNO"]).strip().lower()
                v = str(_mr.get("vendor_item_number", "")).strip()
                if v and v.lower() not in ("nan", "none", ""):
                    sage_vendor_ino[k] = v

        # Source 2 — PO sheet fallback (strip embedded text on the fly)
        if po is not None and "vendor_item_number" in po.columns and "ITEMNO" in po.columns:
            for _, row in po.dropna(subset=["ITEMNO", "vendor_item_number"]).iterrows():
                k = str(row["ITEMNO"]).strip().lower()
                if k in sage_vendor_ino:
                    continue                          # master already has it
                raw = str(row["vendor_item_number"]).strip()
                m = _vin_pat.match(raw)
                v = m.group(1) if m else ""
                if v and v.lower() not in ("nan", "none", ""):
                    sage_vendor_ino[k] = v

        # Pre-filter vendor frames to rows with valid prices
        cpl_price_col = None
        cpl_filt = None
        if cpl is not None and "item_description" in cpl.columns:
            cpl_price_col = "pt_cpl35_price" if "pt_cpl35_price" in cpl.columns else (
                "pt_current_price" if "pt_current_price" in cpl.columns else None)
            if cpl_price_col:
                cpl_filt = cpl[pd.to_numeric(cpl[cpl_price_col], errors="coerce") > 0].copy()

        metabo_filt = None
        if metabo is not None and "item_description" in metabo.columns and "vendor_price_usd" in metabo.columns:
            metabo_filt = metabo[pd.to_numeric(metabo["vendor_price_usd"], errors="coerce") > 0].copy()

        ronix_filt = None
        if ronix is not None and "item_description" in ronix.columns:
            r_pcol = "fob_unit_price_usd" if "fob_unit_price_usd" in ronix.columns else None
            if r_pcol:
                ronix_filt = ronix[pd.to_numeric(ronix[r_pcol], errors="coerce") > 0].copy()

        # Category selector
        if "Sub Category Name" in master.columns:
            pt_categories = sorted([c for c in master["Sub Category Name"].dropna().unique()
                                     if pd.notna(c) and str(c).strip()])
        else:
            pt_categories = []

        if pt_categories:
            selected_category = st.selectbox(
                "Select Performance Tools Category:",
                options=pt_categories,
                key="vendor_item_cat"
            )

            pt_items_in_cat = master[
                (master["Sub Category Name"] == selected_category) &
                (pd.to_numeric(master["performance_tool_price"], errors="coerce") > 0)
            ][["ITEMNO", "item_description", "performance_tool_price"]].drop_duplicates("ITEMNO").head(30)

            if not pt_items_in_cat.empty:
                st.markdown(f"**{selected_category}** — {len(pt_items_in_cat)} items (showing up to 30)")

                chart_rows = []

                for _, pt_row in pt_items_in_cat.iterrows():
                    sage_itemno = str(pt_row["ITEMNO"]).strip()
                    sage_key    = sage_itemno.lower()
                    pt_desc     = str(pt_row.get("item_description", ""))
                    pt_price    = float(pd.to_numeric(pt_row["performance_tool_price"], errors="coerce") or 0)

                    def _match_vendor(vendor_df, price_col):
                        """Three-pass match:
                        Pass 1 — Sage ITEMNO       vs vendor item_number  (>=80%)
                        Pass 2 — vendor_item_number vs vendor item_number  (>=80%)
                        Pass 3 — PT description    vs vendor description   (>=55%)
                        """
                        if vendor_df is None or vendor_df.empty:
                            return None, 0

                        # Pass 1 — Sage ITEMNO vs vendor item_number
                        row1, sc1 = best_itemno_match(sage_itemno, vendor_df)
                        if sc1 >= FUZZY_HIGH:
                            p = pd.to_numeric(row1.get(price_col, None), errors="coerce")
                            return (float(p) if pd.notna(p) and p > 0 else None), round(sc1, 1)

                        # Pass 2 — Sage vendor_item_number vs vendor item_number
                        vin = sage_vendor_ino.get(sage_key, "")
                        if vin:
                            row2, sc2 = best_itemno_match(vin, vendor_df)
                            if sc2 >= FUZZY_HIGH:
                                p = pd.to_numeric(row2.get(price_col, None), errors="coerce")
                                return (float(p) if pd.notna(p) and p > 0 else None), round(sc2, 1)

                        # Pass 3 — description match (catches Metabo/Ronix whose
                        #           item numbers don't overlap with PT codes)
                        vidx = _build_vendor_index(vendor_df, price_col)
                        p3, sc3 = best_desc_match(pt_desc, vidx, threshold=55.0)
                        if p3 is not None:
                            return p3, sc3

                        return None, 0

                    cpl_price,    cpl_score    = _match_vendor(cpl_filt, cpl_price_col) if cpl_filt is not None else (None, 0)
                    metabo_price, metabo_score = _match_vendor(metabo_filt, "vendor_price_usd") if metabo_filt is not None else (None, 0)
                    ronix_price,  ronix_score  = _match_vendor(ronix_filt, "fob_unit_price_usd") if ronix_filt is not None else (None, 0)

                    if any([cpl_price, metabo_price, ronix_price]):
                        label = (pt_desc[:35] + "…") if len(pt_desc) > 35 else pt_desc
                        if pt_price:
                            chart_rows.append({"Item": label, "Vendor": "Performance Tools", "Cost (USD)": pt_price})
                        if cpl_price:
                            chart_rows.append({"Item": label, "Vendor": "CPL", "Cost (USD)": cpl_price})
                        if metabo_price:
                            chart_rows.append({"Item": label, "Vendor": "Metabo", "Cost (USD)": metabo_price})
                        if ronix_price:
                            chart_rows.append({"Item": label, "Vendor": "Ronix (FOB)", "Cost (USD)": ronix_price})

                # --- bar chart only (no comparison table) ---
                if chart_rows:
                    st.markdown("#### Performance Tools vs Matched Vendors — Cost by Item")
                    st.markdown(
                        "<div class='chart-desc'><strong>What this shows:</strong> "
                        "For each item that had at least one vendor match ≥80%, the grouped bars compare "
                        "the current PT cost against the matched vendor's price. "
                        "Ronix prices are FOB (pre-freight). Fewer / shorter bars = potential savings.</div>",
                        unsafe_allow_html=True
                    )
                    df_chart = pd.DataFrame(chart_rows)
                    vendor_colors = {
                        "Performance Tools": "#2563eb",
                        "CPL":          "#f59e0b",
                        "Metabo":       "#10b981",
                        "Ronix (FOB)":  "#ef4444",
                    }
                    fig = px.bar(
                        df_chart,
                        x="Item",
                        y="Cost (USD)",
                        color="Vendor",
                        barmode="group",
                        color_discrete_map=vendor_colors,
                        text="Cost (USD)",
                    )
                    fig.update_traces(
                        texttemplate="$%{text:.2f}",
                        textposition="outside",
                        marker_line_width=0,
                    )
                    fig.update_yaxes(tickprefix="$", tickformat=",.2f")
                    fig.update_xaxes(tickangle=-30)
                    _pd = plotly_defaults()
                    _pd["legend"] = dict(
                        orientation="h",
                        yanchor="bottom", y=1.02,
                        xanchor="right",  x=1,
                        font=dict(color="#d4d0c8"),
                    )
                    fig.update_layout(**_pd, height=420)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No vendor matches ≥80% found for items in this category. Try a different category.")
            else:
                st.info(f"No items found in category '{selected_category}' with valid costs.")
        else:
            st.info("No categories available in Performance Tools data.")


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    data = load_data()

    if not data:
        st.markdown(
            "<div class='context-box'><h4>Warning: No data found</h4>"
            "<p>Run load_vendor_data.py first to generate the processed/ CSV files, "
            "then relaunch this dashboard.</p></div>",
            unsafe_allow_html=True
        )
        return

    page, w_move, w_prof, w_conv, ronix_ff = sidebar(data)

    with st.spinner("Building scores…"):
        scores = build_scores(data, w_move, w_prof, w_conv)

    with st.spinner("Running predictions…"):
        preds = build_predictions(data)

    dispatch = {
        "Overview":             lambda: page_overview(data, scores),
        "Scoring & Ranking":    lambda: page_scoring(data, scores),
        "Predictions & PO Timing": lambda: page_predictions(data, preds, scores),
        "Price Analysis":       lambda: page_price_analysis(data, ronix_ff),
        "Item Deep Dive":       lambda: page_item_deep_dive(data, scores, preds),
        "Vendor Comparison":    lambda: page_vendor_comparison(data, scores),
    }
    dispatch.get(page, dispatch["Overview"])()


if __name__ == "__main__":
    main()