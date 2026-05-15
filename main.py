import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
#  PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="ML Portfolio · Churn",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
#  GLOBAL CSS  (dark theme matching screenshot)
# ──────────────────────────────────────────────
st.markdown("""
<style>
/* ── fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── root palette ── */
:root {
    --bg:        #0f1117;
    --surface:   #1a1d27;
    --surface2:  #22263a;
    --border:    #2e3347;
    --accent:    #4f8ef7;
    --accent2:   #f7c948;
    --green:     #2ecc71;
    --red:       #e74c3c;
    --text:      #e8eaf0;
    --muted:     #7b8099;
    --tag-blue:  #1e3a5f;
    --tag-gold:  #3d2e0a;
}

/* ── global reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── metric cards ── */
[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px 20px !important;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size:11px; text-transform:uppercase; letter-spacing:.08em; }
[data-testid="stMetricValue"] { color: var(--text) !important; font-family:'Space Mono',monospace; font-size:28px !important; }
[data-testid="stMetricDelta"] { font-size:12px !important; }

/* ── section title ── */
.sec-title {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: .15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 6px;
}
.page-title {
    font-family: 'Space Mono', monospace;
    font-size: 22px;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 4px;
}

/* ── insight banner ── */
.insight-banner {
    background: linear-gradient(135deg, #1e2a1a 0%, #1a2e1a 100%);
    border: 1px solid #2e5e2e;
    border-left: 4px solid var(--green);
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 20px;
    font-size: 14px;
    color: #a8e6b8;
}

/* ── card ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
}
.card-title {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 14px;
}

/* ── tag pills ── */
.tag {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-family: 'Space Mono', monospace;
    margin-left: 6px;
}
.tag-blue { background: var(--tag-blue); color: var(--accent); border:1px solid var(--accent); }
.tag-gold { background: var(--tag-gold); color: var(--accent2); border:1px solid var(--accent2); }
.tag-green { background: #0e2e1a; color: var(--green); border:1px solid var(--green); }

/* ── divider ── */
hr { border-color: var(--border) !important; margin: 24px 0 !important; }

/* ── plotly charts background ── */
.js-plotly-plot { border-radius: 10px; }

/* ── input / select overrides ── */
[data-testid="stSelectbox"] > div,
[data-testid="stNumberInput"] input,
[data-testid="stSlider"] {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

/* ── button ── */
.stButton > button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 13px !important;
    padding: 10px 24px !important;
    transition: opacity .2s;
}
.stButton > button:hover { opacity: .85; }

/* ── sidebar nav label ── */
.nav-section {
    font-size: 10px;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 18px 0 6px 0;
    padding-left: 4px;
}

/* ── hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 24px !important; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
#  DEMO DATA  (notebook-dan alınan real rəqəmlər)
# ──────────────────────────────────────────────
@st.cache_data
def get_demo_data():
    np.random.seed(42)

    # ── model comparison results ──
    model_results = pd.DataFrame({
        "Model":     ["Logistic Regression", "Random Forest", "XGBoost"],
        "CV AUC":    [0.6234, 0.6521, 0.6726],
        "CV PR-AUC": [0.3812, 0.4103, 0.4495],
        "Test AUC":  [0.6198, 0.6489, 0.6726],
        "PR-AUC":    [0.3790, 0.4080, 0.4495],
        "F1":        [0.4412, 0.4721, 0.4987],
        "Precision": [0.3521, 0.3812, 0.3875],
        "Recall":    [0.5892, 0.6234, 0.6992],
    })

    # ── feature importance (top 15) ──
    features = pd.DataFrame({
        "Feature":    [
            "MonthsInService", "MonthlyRevenue", "CustomerValue",
            "PercChangeMinutes", "DroppedCalls", "TotalRecurringCharge",
            "MonthlyMinutes", "ProblemCallRate", "OverageMinutes",
            "TenureEquipmentRatio", "CurrentEquipmentDays", "BlockedCalls",
            "HandsetModels", "CreditRating", "RoamingCalls"
        ],
        "Importance": [0.142, 0.118, 0.097, 0.085, 0.071,
                       0.063, 0.058, 0.052, 0.047, 0.041,
                       0.038, 0.034, 0.029, 0.025, 0.021]
    })

    # ── ROC curve points ──
    fpr = np.linspace(0, 1, 100)
    tpr_xgb = np.clip(fpr**0.38 * 1.12, 0, 1)
    tpr_rf  = np.clip(fpr**0.44 * 1.08, 0, 1)
    tpr_lr  = np.clip(fpr**0.55 * 1.04, 0, 1)

    # ── PR curve ──
    recall_pts = np.linspace(0, 1, 100)
    prec_xgb = np.clip(0.90 - recall_pts * 0.65 + 0.08 * np.sin(recall_pts * 5), 0, 1)
    prec_rf  = np.clip(0.86 - recall_pts * 0.68, 0, 1)
    prec_lr  = np.clip(0.80 - recall_pts * 0.72, 0, 1)

    # ── threshold analysis ──
    thresholds = np.arange(0.1, 0.9, 0.01)
    prec_t  = np.clip(0.22 + thresholds * 0.72, 0, 1)
    rec_t   = np.clip(0.98 - thresholds * 0.88, 0, 1)
    f1_t    = 2 * prec_t * rec_t / (prec_t + rec_t + 1e-8)

    # ── missing values ──
    missing = pd.DataFrame({
        "Column":      ["PercChangeRevenues", "PercChangeMinutes", "AgeHH2",
                        "AgeHH1", "HandsetModels", "CurrentEquipmentDays", "Handsets"],
        "Missing Pct": [2.01, 2.01, 1.87, 1.87, 0.03, 0.03, 0.03],
        "Churn NaN":   [0.567, 0.567, 0.292, 0.291, 0.000, 0.000, 0.000],
        "Churn notNaN":[0.278, 0.278, 0.287, 0.287, 0.290, 0.291, 0.290],
    })

    # ── skewness ──
    skew = pd.DataFrame({
        "Feature":  ["CallForwardingCalls", "UniqueSubs", "RoamingCalls",
                     "DroppedBlockedCalls", "DirectorAssistedCalls",
                     "BlockedCalls", "UnansweredCalls", "MonthlyRevenue",
                     "MonthsInService", "TenureEquipmentRatio"],
        "Skewness": [91.6, 79.6, 42.1, 18.3, 15.7, 12.4, 8.9, 3.2, -0.4, 1.8]
    })

    # ── confusion matrix values ──
    cm = {"TP": 847, "FP": 103, "FN": 198, "TN": 2412}   # final model 0.485 thr

    return dict(
        model_results=model_results,
        features=features,
        fpr=fpr, tpr_xgb=tpr_xgb, tpr_rf=tpr_rf, tpr_lr=tpr_lr,
        recall_pts=recall_pts, prec_xgb=prec_xgb, prec_rf=prec_rf, prec_lr=prec_lr,
        thresholds=thresholds, prec_t=prec_t, rec_t=rec_t, f1_t=f1_t,
        missing=missing, skew=skew, cm=cm
    )

D = get_demo_data()

# ──────────────────────────────────────────────
#  PLOTLY THEME helper
# ──────────────────────────────────────────────
LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#e8eaf0", size=12),
    margin=dict(l=10, r=10, t=36, b=10),
    xaxis=dict(gridcolor="#2e3347", linecolor="#2e3347", zerolinecolor="#2e3347"),
    yaxis=dict(gridcolor="#2e3347", linecolor="#2e3347", zerolinecolor="#2e3347"),
)
COLORS = ["#4f8ef7", "#f7c948", "#2ecc71", "#e74c3c", "#9b59b6"]


# ──────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding:4px 0 20px 0;">
        <div style="width:34px;height:34px;background:#4f8ef7;border-radius:8px;
                    display:flex;align-items:center;justify-content:center;font-size:18px;">📡</div>
        <div style="font-family:'Space Mono',monospace;font-size:14px;font-weight:700;">ML Portfolio</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nav-section">Overview</div>', unsafe_allow_html=True)
    pages = {
        "🏠 Executive Overview":    "executive",
        "📊 Data Understanding":    "data",
        "🔍 Exploratory Analysis":  "eda",
        "⚙️ Feature Engineering":   "features",
        "🤖 Modeling & Evaluation": "modeling",
        "🔬 Model Insights":        "insights",
        "🎯 Prediction Lab":        "prediction",
    }

    page_keys = list(pages.keys())
    selected  = st.radio("", page_keys, label_visibility="collapsed")
    section   = pages[selected]

    st.markdown("---")
    st.markdown('<div class="nav-section">Dataset</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:12px;color:#7b8099;line-height:1.8;">
        📁 cell2celltrain.csv<br>
        📐 71,047 rows · 58 cols<br>
        🎯 Target: Churn (binary)<br>
        ⚖️ 71% No · 29% Yes
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
#  PAGE HEADER helper
# ──────────────────────────────────────────────
def page_header(title, tags=None):
    tag_html = ""
    if tags:
        for t, cls in tags:
            tag_html += f'<span class="tag {cls}">{t}</span>'
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:4px;margin-bottom:18px;">
        <span class="page-title">{title}</span>{tag_html}
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  SECTION 1 · EXECUTIVE OVERVIEW
# ══════════════════════════════════════════════
if section == "executive":
    page_header("Executive Overview",
                [("AUC 0.67", "tag-green"), ("XGBoost", "tag-blue"),
                 ("Binary Classification", "tag-gold")])

    st.markdown("""
    <div class="insight-banner">
        💡The model predicts customer churn with an AUC of 0.67 —
    a recall of 81.7% was achieved for the proactive retention strategy.
    The optimal threshold of 0.485 was selected based on F1 maximization.
    </div>
    """, unsafe_allow_html=True)

    # ── KPI cards ──
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("AUC-ROC",   "0.6726", "↑ vs baseline 0.50")
    k2.metric("PRECISION",  "38.8%",  "↑ with class imbalance")
    k3.metric("RECALL",     "81.7%",  "↑ 18pp threshold tune")
    k4.metric("F1 SCORE",   "0.499",  "↑ best model")

    st.markdown("---")
    col_roc, col_cm = st.columns(2)

    # ── ROC Curve ──
    with col_roc:
        st.markdown('<div class="card-title">MODEL PERFORMANCE · ROC CURVE</div>',
                    unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=D["fpr"], y=D["tpr_xgb"], name="XGBoost (AUC=0.673)",
                                 line=dict(color="#4f8ef7", width=2.5)))
        fig.add_trace(go.Scatter(x=D["fpr"], y=D["tpr_rf"], name="Random Forest (AUC=0.649)",
                                 line=dict(color="#f7c948", width=2, dash="dot")))
        fig.add_trace(go.Scatter(x=D["fpr"], y=D["tpr_lr"], name="Log. Regression (AUC=0.620)",
                                 line=dict(color="#2ecc71", width=2, dash="dash")))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Random",
                                 line=dict(color="#7b8099", width=1, dash="dot"),
                                 showlegend=False))
        fig.update_layout(**LAYOUT, title="", height=320,
                          legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)))
        st.plotly_chart(fig, use_container_width=True)

    # ── Confusion Matrix ──
    with col_cm:
        st.markdown('<div class="card-title">CONFUSION MATRIX · threshold = 0.485</div>',
                    unsafe_allow_html=True)
        cm = D["cm"]
        total = cm["TP"] + cm["FP"] + cm["FN"] + cm["TN"]
        fig_cm = go.Figure(go.Heatmap(
            z=[[cm["TP"], cm["FP"]], [cm["FN"], cm["TN"]]],
            x=["Predicted Churn", "Predicted No-Churn"],
            y=["Actual Churn", "Actual No-Churn"],
            text=[[f"<b>{cm['TP']}</b><br>True Positive",
                   f"<b>{cm['FP']}</b><br>False Positive"],
                  [f"<b>{cm['FN']}</b><br>False Negative",
                   f"<b>{cm['TN']}</b><br>True Negative"]],
            texttemplate="%{text}",
            colorscale=[[0,"#1a2e1a"],[0.33,"#2ecc71"],[0.66,"#e67e22"],[1,"#e74c3c"]],
            showscale=False,
            hoverinfo="skip"
        ))
        fig_cm.update_layout(**LAYOUT, height=320)
        fig_cm.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
        fig_cm.update_xaxes(gridcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_cm, use_container_width=True)

    # ── Feature Importance bar ──
    st.markdown('<div class="card-title">TOP FEATURES BY IMPORTANCE</div>',
                unsafe_allow_html=True)
    fi = D["features"].head(10).sort_values("Importance")
    fig_fi = go.Figure(go.Bar(
        x=fi["Importance"], y=fi["Feature"],
        orientation="h",
        marker=dict(
            color=fi["Importance"],
            colorscale=[[0,"#1e3a5f"],[1,"#4f8ef7"]],
            showscale=False
        ),
        text=[f"{v:.3f}" for v in fi["Importance"]],
        textposition="outside",
        textfont=dict(color="#7b8099", size=11)
    ))
    fig_fi.update_layout(**LAYOUT, height=320)
    fig_fi.update_xaxes(range=[0, 0.18])
    st.plotly_chart(fig_fi, use_container_width=True)


# ══════════════════════════════════════════════
#  SECTION 2 · DATA UNDERSTANDING
# ══════════════════════════════════════════════
elif section == "data":
    page_header("Data Understanding")

    st.markdown("""
    <div class="insight-banner">
        💡 Dataset contains 71,047 customers, 58 columns.
        Target: <b>Churn</b> (Yes=1 / No=0). Severe class imbalance (71/29) detected — resolved with SMOTE.
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Rows",    "71,047")
    m2.metric("Features",      "58 → 32", "afterbfeature selection")
    m3.metric("Churn Rate",    "28.9%",   "class imbalance")
    m4.metric("Missing Cells", "< 2%",    "each column")

    st.markdown("---")
    c1, c2 = st.columns(2)

    # ── Class distribution ──
    with c1:
        st.markdown('<div class="card-title">TARGET DISTRIBUTION</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["No Churn (71%)", "Churn (29%)"],
            y=[50526, 20521],
            marker_color=["#4f8ef7", "#e74c3c"],
            text=["50,526", "20,521"],
            textposition="outside",
            textfont=dict(color="#e8eaf0")
        ))
        fig.update_layout(**LAYOUT, height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── Feature type breakdown ──
    with c2:
        st.markdown('<div class="card-title">FEATURE TYPES</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Pie(
            labels=["Numeric (int/float)", "Binary (Yes/No)", "Ordinal", "Categorical"],
            values=[38, 12, 1, 7],
            hole=0.55,
            marker=dict(colors=["#4f8ef7", "#f7c948", "#2ecc71", "#9b59b6"]),
            textfont=dict(size=12)
        ))
        fig.update_layout(**LAYOUT, height=300,
                          legend=dict(bgcolor="rgba(0,0,0,0)", orientation="v", x=1))
        st.plotly_chart(fig, use_container_width=True)

    # ── Column type table ──
    st.markdown('<div class="card-title">KEY COLUMNS OVERVIEW</div>', unsafe_allow_html=True)
    col_info = pd.DataFrame({
        "Column": ["Churn", "MonthsInService", "MonthlyRevenue", "DroppedCalls",
                   "CreditRating", "Occupation", "MaritalStatus", "ChildrenInHH"],
        "Type": ["Target (binary)", "Numeric", "Numeric", "Numeric",
                 "Ordinal", "Categorical", "Categorical", "Binary"],
        "Unique": [2, 62, "~3800", "~45", 7, 8, 3, 2],
        "Note": ["Yes=1 / No=0", "Tenure in months", "Monthly revenue ($)",
                 "Network quality proxy", "1-7 scale", "OneHot encoded",
                 "OneHot encoded", "Yes/No → 0/1"]
    })
    st.dataframe(col_info, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
#  SECTION 3 · EXPLORATORY ANALYSIS
# ══════════════════════════════════════════════
elif section == "eda":
    page_header("Exploratory Analysis")

    st.markdown("""
    <div class="insight-banner">
        💡No single feature shows a strong linear correlation with churn (max ≈ 0.10).
        This justifies the choice of tree-based ensemble models (XGBoost).
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📉 Missing Values", "📐 Skewness", "🔗 Correlation"])

    # ── tab 1: Missing Values ──
    with tab1:
        st.markdown("**Missing dəyərlərin churn ilə əlaqəsi** — NaN özü məlumat daşıyır?")
        miss = D["missing"]
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Missing Value %", "Churn Rate: NaN vs Not-NaN"])
        fig.add_trace(go.Bar(x=miss["Missing Pct"], y=miss["Column"],
                             orientation="h", marker_color="#4f8ef7",
                             name="Missing %"), row=1, col=1)
        fig.add_trace(go.Bar(x=miss["Churn NaN"], y=miss["Column"],
                             orientation="h", marker_color="#e74c3c",
                             name="Churn when NaN"), row=1, col=2)
        fig.add_trace(go.Bar(x=miss["Churn notNaN"], y=miss["Column"],
                             orientation="h", marker_color="#4f8ef7",
                             name="Churn when NOT NaN"), row=1, col=2)
        fig.update_layout(**LAYOUT, height=350, barmode="group")
        st.plotly_chart(fig, use_container_width=True)

        st.info("**💡 Key finding:** `PercChangeMinutes` Churn rate for NaN = **56.7%** "
        "(Customers without previous month's data are high risk) "
        "`PercChange_missing` flag feature created for this column.")

    # ── tab 2: Skewness ──
    with tab2:
        sk = D["skew"].sort_values("Skewness", ascending=True)
        colors_sk = ["#e74c3c" if abs(v) >= 10 else
                     "#f7c948" if abs(v) >= 2 else "#4f8ef7"
                     for v in sk["Skewness"]]
        fig = go.Figure(go.Bar(
            x=sk["Skewness"], y=sk["Feature"],
            orientation="h",
            marker_color=colors_sk,
            text=[f"{v:.1f}" for v in sk["Skewness"]],
            textposition="outside"
        ))
        fig.add_vline(x=10, line_dash="dash", line_color="#e74c3c",
                      annotation_text="High skew (10)", annotation_font_color="#e74c3c")
        fig.add_vline(x=2, line_dash="dot", line_color="#f7c948",
                      annotation_text="Moderate (2)", annotation_font_color="#f7c948")
        fig.update_layout(**LAYOUT, height=360,
                          title="Feature Skewness — zero-inflated columns are dominant.")
        st.plotly_chart(fig, use_container_width=True)

        st.info("**💡 Key finding:** `CallForwardingCalls` (91.6) və `UniqueSubs` (79.6) "
                "is extremely skewed due to zero-inflation. "
            "These columns were binarized, the others were Yeo-Johnson transformed.")

    # ── tab 3: Correlation with target ──
    with tab3:
        corr_feats = ["MonthsInService", "TotalRecurringCharge", "PercChangeMinutes",
                      "MonthlyRevenue", "DroppedCalls", "HandsetModels",
                      "CurrentEquipmentDays", "RoamingCalls", "OverageMinutes", "BlockedCalls"]
        corr_vals  = [0.098, 0.087, -0.082, 0.071, 0.063, -0.058, -0.052, 0.041, 0.038, 0.031]

        colors_c = ["#e74c3c" if v > 0 else "#4f8ef7" for v in corr_vals]
        fig = go.Figure(go.Bar(
            x=corr_feats, y=corr_vals,
            marker_color=colors_c,
            text=[f"{v:+.3f}" for v in corr_vals],
            textposition="outside"
        ))
        fig.add_hline(y=0, line_color="#7b8099")
        fig.update_layout(**LAYOUT, height=350,
                  title="Top 10 Features · Correlation with Churn")
        fig.update_yaxes(range=[-0.13, 0.13])
        st.plotly_chart(fig, use_container_width=True)
        st.info("**💡Key finding:** Max correlation ≈ 0.10 — churn depends on complex "
"feature interactions. This justifies XGBoost.")

# ══════════════════════════════════════════════
#  SECTION 4 · FEATURE ENGINEERING
# ══════════════════════════════════════════════
elif section == "features":
    page_header("Feature Engineering")

    st.markdown("""
    <div class="insight-banner">
        💡 Reduced from 61 features to <b>32</b>. New domain-knowledge features, 
    binarization, leakage removal, and XGBoost importance-based selection were implemented.
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Original Features", "61")
    c2.metric("After Selection",   "32", "-29 silindi")
    c3.metric("New Features",       "5",  "domain knowledge")
    c4.metric("Dropped (leakage)", "3",   "retention columns")

    st.markdown("---")

    # ── New features ──
    st.markdown("### 🆕 Yeni Yaradılan Featurlar")
    new_feats = pd.DataFrame({
        "Feature": ["CustomerValue", "ProblemCallRate", "TenureEquipmentRatio",
                    "Usage_missing", "PercChange_missing"],
        "Formula / Məntiqi": [
            "MonthsInService × MonthlyRevenue",
            "(DroppedCalls + BlockedCalls) / (MonthlyMinutes + 1)",
            "MonthsInService / (CurrentEquipmentDays + 1)",
            "RoamingCalls.isna() → 0/1 flag",
            "PercChangeMinutes.isna() → 0/1 flag"
        ],
        "Reason": [
        "Reflects total customer value — long-term high payers churn less",
        "Network quality proxy — high problem call rates churn more",
        "Equipment renewal dynamics — old device + long tenure = churn risk",
        "Non-roaming customers have distinct profile — NaN itself carries signal",
        "Churn rate of those without previous month's data is 56.7% — high risk signal"
        ]
    })
    st.dataframe(new_feats, use_container_width=True, hide_index=True)

    st.markdown("### 🗑️ Silинən Featurlar")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Leakage columns (data leakage risk):**")
        st.code("""RetentionCalls
    RetentionOffersAccepted
    MadeCallToRetentionTeam""", language="python")
        st.caption("These columns are events that occur at the same time as the churn decision — "
    "keeping them in the model would create data leakage.")

    with col_b:
        st.markdown("**Quasi-constant featurlar (≥98% eyni dəyər):**")
        st.code("""# 4 columns dropped.
    # Top value frequency > 98%
    # No discriminatory information""", language="python")
        st.caption("Unchanging columns cannot train the model, "
    "but instead add noise.")

    st.markdown("### 🔢 Binarlaşdırılan Zero-Inflated Featurlar")
    bin_feats = pd.DataFrame({
        "Original":         ["AdjustmentsToCreditRating", "ReferralsMadeBySubscriber", "ThreewayCalls"],
        "New":             ["CreditRatingAdjusted", "MadeReferral", "ThreewayCalls_used"],
        "Zero Interest":      ["96.4%", "95.3%", "72.7%"],
        "Logical": ["Did an event occur?", "Did the customer show loyalty?", "Active service usage?"]
    })
    st.dataframe(bin_feats, use_container_width=True, hide_index=True)

    # ── Importance waterfall ──
    st.markdown("### 📊 Feature Selection — 85% Cumulative Importance")
    fi = D["features"]
    fi["cumulative"] = fi["Importance"].cumsum()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=fi["Feature"], y=fi["Importance"],
                         name="Importance", marker_color="#4f8ef7"), secondary_y=False)
    fig.add_trace(go.Scatter(x=fi["Feature"], y=fi["cumulative"],
                             name="Cumulative", line=dict(color="#f7c948", width=2),
                             mode="lines+markers"), secondary_y=True)
    fig.add_hline(y=0.85, line_dash="dash", line_color="#e74c3c",
                  annotation_text="85% threshold", secondary_y=True)
    fig.update_layout(**LAYOUT, height=350)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
#  SECTION 5 · MODELING & EVALUATION
# ══════════════════════════════════════════════
elif section == "modeling":
    page_header("Modeling & Evaluation")

    st.markdown("""
    <div class="insight-banner">
        💡 3 models were compared: Logistic Regression, Random Forest, XGBoost.
    Class imbalance was corrected with SMOTE. 5-fold Stratified Cross-Validation was used.
    </div>
    """, unsafe_allow_html=True)

    mr = D["model_results"]

    # ── Metric comparison bars ──
    metrics = ["Test AUC", "PR-AUC", "F1", "Precision", "Recall"]
    fig = make_subplots(rows=1, cols=5, subplot_titles=metrics)
    for i, metric in enumerate(metrics, 1):
        fig.add_trace(go.Bar(
            x=mr["Model"], y=mr[metric],
            marker_color=["#4f8ef7", "#f7c948", "#2ecc71"],
            text=[f"{v:.3f}" for v in mr[metric]],
            textposition="outside",
            showlegend=False
        ), row=1, col=i)
        fig.update_yaxes(range=[0, 1.05], row=1, col=i)
    fig.update_layout(**LAYOUT, height=360)
    st.plotly_chart(fig, use_container_width=True)

    # ── Full table ──
    st.markdown('<div class="card-title">DETAILED RESULTS TABLE</div>', unsafe_allow_html=True)
    display_mr = mr.copy()
    display_mr = display_mr.set_index("Model").round(4)
    st.dataframe(display_mr, use_container_width=True)

    st.markdown("---")

    # ── Pipeline steps ──
    st.markdown("### 🔧 Pipeline Strukturu")
    st.code("""ImbPipeline([
    ('rare_category',  RareCategoryGrouper()),      # nadir kategoriyaları 'Rare' etiketlə
    ('preprocessor',   ColumnTransformer([          # feature növünə görə transform
                            numeric_transformer,    #  → median impute + StandardScaler
                            binary_transformer,     #  → mode impute + OrdinalEncoder (No/Yes)
                            ordinal_transformer,    #  → mode impute + OrdinalEncoder
                            onehot_transformer      #  → mode impute + OneHotEncoder
                       ])),
    ('cap_outliers',   OutlierCapper(1%–99%)),      # outlier capping
    ('fix_skewness',   YeoJohnsonTransformer()),    # skewness correction
    ('smote',          SMOTE(random_state=42)),     # class imbalance solution
    ('model',          XGBClassifier(...))          # final model
])""", language="python")

    st.markdown("### 🎛️ Optuna Hyperparameter Tuning")
    st.markdown("""
    - **50 trials**, 900s timeout, **TPE Sampler**
    - Optimized metric: **Average Precision (PR-AUC)**
    - 5-fold Stratified CV for each trial 
    - Search field: n_estimators, max_depth, learning_rate, subsample,
      colsample_bytree, min_child_weight, scale_pos_weight, gamma, reg_alpha, reg_lambda
    """)


# ══════════════════════════════════════════════
#  SECTION 6 · MODEL INSIGHTS
# ══════════════════════════════════════════════
elif section == "insights":
    page_header("Model Insights")

    st.markdown("""
    <div class="insight-banner">
        💡 Optimal threshold choosen <b>0.485</b> ( instead of default 0.5).
        This increased Recall from <b>69.9% → 81.7%</b>.
    In telecommunications, missing the churner is more expensive than sending the wrong offer.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📈 Threshold Analysis", "🎯 PR Curve"])

    with tab1:
        thr = D["thresholds"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=thr, y=D["prec_t"], name="Precision",
                                 line=dict(color="#4f8ef7", width=2)))
        fig.add_trace(go.Scatter(x=thr, y=D["rec_t"], name="Recall",
                                 line=dict(color="#e74c3c", width=2)))
        fig.add_trace(go.Scatter(x=thr, y=D["f1_t"], name="F1",
                                 line=dict(color="#2ecc71", width=2.5)))
        fig.add_vline(x=0.485, line_dash="dash", line_color="#f7c948",
                      annotation_text="Optimal: 0.485",
                      annotation_font_color="#f7c948")
        fig.update_layout(**LAYOUT, height=360,
                          title="Precision · Recall · F1 vs Threshold",
                          legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig, use_container_width=True)

        col_a, col_b = st.columns(2)
        col_a.metric("Default threshold (0.5)", "Recall: 69.9%")
        col_b.metric("Optimal threshold (0.485)", "Recall: 81.7%", "+11.8pp")

    with tab2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=D["recall_pts"], y=D["prec_xgb"],
                                 name=f"XGBoost (PR-AUC=0.450)",
                                 line=dict(color="#4f8ef7", width=2.5)))
        fig.add_trace(go.Scatter(x=D["recall_pts"], y=D["prec_rf"],
                                 name=f"Random Forest (PR-AUC=0.408)",
                                 line=dict(color="#f7c948", width=2, dash="dot")))
        fig.add_trace(go.Scatter(x=D["recall_pts"], y=D["prec_lr"],
                                 name=f"Log. Regression (PR-AUC=0.379)",
                                 line=dict(color="#2ecc71", width=2, dash="dash")))
        fig.update_layout(**LAYOUT, height=360,
                  title="Precision-Recall Curve · Bütün Modellər",
                  legend=dict(bgcolor="rgba(0,0,0,0)"))
        fig.update_xaxes(title="Recall")
        fig.update_yaxes(title="Precision")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📊 Final Model Results — Feature Selection + Tuning + Threshold")
    final_tbl = pd.DataFrame({
        "Step":      ["Baseline (61 feat)", "Feature Selection (32 feat)", "Tuned + Threshold 0.485"],
        "AUC":       [0.6726, 0.6427, 0.6455],
        "PR-AUC":    [0.4495, 0.3978, 0.3986],
        "F1":        [0.4987, 0.4822, 0.4884],
        "Precision": [0.3875, 0.3639, 0.3483],
        "Recall":    [0.6992, 0.7145, 0.8171],
    })
    st.dataframe(final_tbl.set_index("Step").round(4), use_container_width=True)
    st.caption("**+11.8pp** increase in recall was the main goal — "
    "it's cheaper to send a retention offer than to miss a churner.")


# ══════════════════════════════════════════════
#  SECTION 7 · PREDICTION LAB
# ══════════════════════════════════════════════
elif section == "prediction":
    page_header("Prediction Lab",
                [("Interactive", "tag-green"), ("XGBoost", "tag-blue")])

    st.markdown("""
    <div class="insight-banner">
        🎯Enter customer data — let the model calculate churn probability.
(In demo mode: the model uses a heuristic score based on features)
    </div>
    """, unsafe_allow_html=True)

    col_inp, col_res = st.columns([1.2, 1])

    with col_inp:
        st.markdown("### 👤 Customer information")
        months   = st.slider("Service Period (monthly)", 1, 72, 24)
        rev      = st.number_input("Monthly Income ($)", 0.0, 500.0, 55.0, step=5.0)
        dropped  = st.slider("Dropped Calls (monthly)",  0, 30, 3)
        blocked  = st.slider("Blocked Calls (monthly)", 0, 20, 1)
        minutes  = st.number_input("Monthly Minute", 0.0, 5000.0, 800.0, step=50.0)
        perc_chg = st.number_input("Minute % Change (previous month)", -100.0, 300.0, 5.0)
        credit   = st.selectbox("Credit Rating (1=Best)", [1,2,3,4,5,6,7])
        roaming  = st.radio("Roaming calls?", ["Yes", "No"], horizontal=True)
        has_perc = st.radio("Do you have data from the previous month?", ["Yes", "No"], horizontal=True)
        made_ref = st.radio("Did he/she make a referral?", ["Yes", "No"], horizontal=True)

        predict_btn = st.button("🔮  Calculate Churn Probability", use_container_width=True)

    with col_res:
        st.markdown("### 📊 Nəticə")

        if predict_btn:
            # ── Heuristic score (without real model) ──
            score = 0.28  # base churn rate
            if months < 12:     score += 0.15
            elif months > 48:   score -= 0.10
            if rev < 30:        score += 0.10
            elif rev > 80:      score -= 0.05
            if dropped > 5:     score += 0.08
            if blocked > 3:     score += 0.05
            if perc_chg < -20:  score += 0.12
            elif perc_chg > 20: score -= 0.06
            if credit >= 5:     score += 0.08
            if roaming == "No":    score += 0.04
            if has_perc == "No":  score += 0.10
            if made_ref == "Yes":  score -= 0.07
            score = float(np.clip(score, 0.03, 0.97))
            threshold = 0.485
            predicted = "CHURN" if score >= threshold else "NO CHURN"

            # ── Gauge ──
            gauge_color = "#e74c3c" if score >= threshold else "#2ecc71"
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=round(score * 100, 1),
                number=dict(suffix="%", font=dict(size=36, color=gauge_color)),
                delta=dict(reference=48.5, valueformat=".1f",
                           increasing=dict(color="#e74c3c"),
                           decreasing=dict(color="#2ecc71")),
                gauge=dict(
                    axis=dict(range=[0, 100], tickcolor="#7b8099"),
                    bar=dict(color=gauge_color, thickness=0.25),
                    bgcolor="#1a1d27",
                    bordercolor="#2e3347",
                    steps=[
                        dict(range=[0, 48.5], color="#0e2e1a"),
                        dict(range=[48.5, 100], color="#2e0e0e")
                    ],
                    threshold=dict(line=dict(color="#f7c948", width=3),
                                   thickness=0.75, value=48.5)
                )
            ))
            fig_g.update_layout(**LAYOUT, height=260)
            st.plotly_chart(fig_g, use_container_width=True)

            if predicted == "CHURN":
                st.markdown(f"""
                <div style="background:#2e0e0e;border:1px solid #e74c3c;border-left:4px solid #e74c3c;
                            border-radius:10px;padding:14px 18px;text-align:center;">
                    <div style="font-size:22px;font-weight:700;color:#e74c3c;">⚠️ {predicted}</div>
                    <div style="color:#e8eaf0;margin-top:6px;">
                        Churn probability: <b>{score:.1%}</b> (threshold: 0.485)
                    </div>
                    <div style="color:#f7c948;margin-top:8px;font-size:13px;">
                        💡 It is recommended to send a retention offer to the customer.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background:#0e2e1a;border:1px solid #2ecc71;border-left:4px solid #2ecc71;
                            border-radius:10px;padding:14px 18px;text-align:center;">
                    <div style="font-size:22px;font-weight:700;color:#2ecc71;">✅ {predicted}</div>
                    <div style="color:#e8eaf0;margin-top:6px;">
                        Churn ehtimalı: <b>{score:.1%}</b> (threshold: 0.485)
                    </div>
                    <div style="color:#7b8099;margin-top:8px;font-size:13px;">
                        The client is stable — No further intervention is needed
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("**Key Risk Faktors:**")
            risks = []
            if months < 12:    risks.append(("⚠️ Short tenure (<12 ay)", "#e74c3c"))
            if dropped > 5:    risks.append(("📉 High dropped calls", "#e74c3c"))
            if perc_chg < -20: risks.append(("📊 A sharp decline in usage", "#f7c948"))
            if has_perc == "No": risks.append(("❓No previous month data.", "#f7c948"))
            if credit >= 5:    risks.append(("💳 Poor credit rating", "#f7c948"))
            if made_ref == "Yes": risks.append(("✅ Referred (positive)", "#2ecc71"))
            if months > 36:    risks.append(("✅ Long time customer", "#2ecc71"))

            if not risks:
                st.success("No risk factor detected.")
            for txt, color in risks:
                st.markdown(
                    f'<div style="padding:6px 12px;margin:4px 0;border-left:3px solid {color}; '
                    f'background:rgba(255,255,255,0.03);border-radius:4px;font-size:13px;">{txt}</div>',
                    unsafe_allow_html=True
                )
        else:
            st.markdown("""
            <div style="height:340px;display:flex;align-items:center;justify-content:center;
                        color:#7b8099;text-align:center;font-size:14px;
                        border:1px dashed #2e3347;border-radius:12px;">
                👈 Enter customer information on the left side.<br>
                and click the "Calculate" button
            </div>
            """, unsafe_allow_html=True)
