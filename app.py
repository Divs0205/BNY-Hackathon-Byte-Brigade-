import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KYC Intelligence Platform",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Sidebar ────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0A0F1E;
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #F8FAFC !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stFileUploader label { color: #94A3B8 !important; font-size: 12px !important; text-transform: uppercase; letter-spacing: 0.08em; }

/* ── Main background ────────────────────── */
.main { background: #F8FAFC; }
[data-testid="stAppViewContainer"] { background: #F0F4F8; }

/* ── KPI card ───────────────────────────── */
.kpi-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 24px 28px;
    border: 1px solid rgba(0,0,0,0.06);
    box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 16px rgba(0,0,0,0.04);
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 16px 16px 0 0;
}
.kpi-card.green::before  { background: linear-gradient(90deg, #22C55E, #16A34A); }
.kpi-card.amber::before  { background: linear-gradient(90deg, #F59E0B, #D97706); }
.kpi-card.red::before    { background: linear-gradient(90deg, #EF4444, #DC2626); }
.kpi-card.blue::before   { background: linear-gradient(90deg, #3B82F6, #2563EB); }
.kpi-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #94A3B8;
    margin-bottom: 10px;
}
.kpi-value {
    font-size: 32px;
    font-weight: 600;
    color: #0F172A;
    line-height: 1;
    margin-bottom: 4px;
    font-variant-numeric: tabular-nums;
}
.kpi-sub {
    font-size: 13px;
    color: #64748B;
    font-family: 'DM Mono', monospace;
}
.kpi-icon {
    position: absolute;
    right: 24px; top: 24px;
    font-size: 28px;
    opacity: 0.15;
}

/* ── Section header ─────────────────────── */
.section-header {
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #64748B;
    margin: 32px 0 16px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #E2E8F0;
}

/* ── Risk badge ─────────────────────────── */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 100px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.badge-high   { background: #FEE2E2; color: #991B1B; }
.badge-medium { background: #FEF3C7; color: #92400E; }
.badge-low    { background: #DCFCE7; color: #166534; }

/* ── Page title ─────────────────────────── */
.page-title {
    font-size: 28px;
    font-weight: 600;
    color: #0F172A;
    letter-spacing: -0.02em;
    margin: 0;
}
.page-subtitle {
    font-size: 14px;
    color: #64748B;
    margin-top: 4px;
    margin-bottom: 0;
}

/* ── Tabs ───────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: #ffffff;
    border-radius: 12px;
    padding: 4px;
    gap: 2px;
    border: 1px solid #E2E8F0;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-size: 13px;
    font-weight: 500;
    color: #64748B;
    padding: 8px 20px;
    border: none;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    background: #0F172A !important;
    color: #ffffff !important;
}

/* ── Chart container ────────────────────── */
.chart-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 24px;
    border: 1px solid rgba(0,0,0,0.06);
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    margin-bottom: 20px;
}
.chart-title {
    font-size: 14px;
    font-weight: 600;
    color: #0F172A;
    margin-bottom: 4px;
}
.chart-subtitle {
    font-size: 12px;
    color: #94A3B8;
    margin-bottom: 16px;
}

/* ── Dataframe ──────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #E2E8F0 !important;
}

/* ── Search bar ─────────────────────────── */
.stTextInput input {
    border-radius: 10px;
    border: 1.5px solid #E2E8F0;
    font-size: 13px;
    background: #ffffff;
}
.stTextInput input:focus {
    border-color: #3B82F6;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.1);
}

/* ── Divider ────────────────────────────── */
hr { border: none; border-top: 1px solid #E2E8F0; margin: 24px 0; }

/* ── Alert box ──────────────────────────── */
.alert-box {
    background: #FFF7ED;
    border: 1px solid #FED7AA;
    border-radius: 12px;
    padding: 16px 20px;
    display: flex;
    align-items: flex-start;
    gap: 12px;
}
.alert-box.critical {
    background: #FEF2F2;
    border-color: #FECACA;
}

/* ── Score bar ──────────────────────────── */
.score-bar-wrap {
    height: 6px;
    background: #F1F5F9;
    border-radius: 100px;
    overflow: hidden;
    margin-top: 6px;
}
.score-bar-fill {
    height: 100%;
    border-radius: 100px;
    transition: width 0.4s ease;
}

/* ── Empty state ────────────────────────── */
.empty-state {
    text-align: center;
    padding: 80px 40px;
    color: #94A3B8;
}
.empty-state h2 { font-size: 20px; font-weight: 500; color: #334155; margin-bottom: 8px; }
.empty-state p  { font-size: 14px; margin: 0; }

/* ── Metric delta ───────────────────────── */
[data-testid="stMetricDelta"] { font-family: 'DM Mono', monospace; font-size: 12px; }

/* ── Scrollbar ──────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 100px; }

/* ── Sidebar status pill ────────────────── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(34,197,94,0.15);
    color: #16A34A;
    font-size: 12px;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 100px;
    margin-top: 8px;
}
.dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }

/* ── Model toggle ───────────────────────── */
[data-testid="stCheckbox"] { font-size: 13px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# COLOR PALETTES
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    'HIGH':   '#EF4444',
    'MEDIUM': '#F59E0B',
    'LOW':    '#22C55E',
    'bg':     '#F0F4F8',
    'card':   '#ffffff',
    'navy':   '#0F172A',
    'slate':  '#64748B',
    'muted':  '#94A3B8',
}

PLOTLY_BASE = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='DM Sans, sans-serif', color='#334155'),
    margin=dict(t=10, b=10, l=10, r=10),
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def kpi_card(label, value, sub, color, icon, col):
    col.markdown(f"""
    <div class="kpi-card {color}">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def score_color(score):
    if score >= 75: return '#EF4444'
    if score >= 40: return '#F59E0B'
    return '#22C55E'


def tier_badge(tier):
    cls = {'HIGH': 'high', 'MEDIUM': 'medium', 'LOW': 'low'}.get(str(tier).upper(), 'low')
    return f'<span class="badge badge-{cls}">{tier}</span>'


def normalize_decision(d):
    d = str(d).upper().strip()
    if 'REJECT' in d or 'EDD' in d:
        return 'REJECT / EDD'
    if 'MANUAL' in d:
        return 'MANUAL REVIEW'
    return 'APPROVE'


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔐 KYC Intelligence")
    st.markdown('<div class="status-pill"><div class="dot"></div>Engine Online</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Upload Output")
    st.caption("Run the KYC engine to produce `kyc_output.csv`, then upload it here.")
    uploaded_file = st.file_uploader("kyc_output.csv", type=["csv"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### Display Options")
    show_rf     = st.checkbox("Show RF vs Rule Score breakdown", value=True)
    show_alerts = st.checkbox("Show high-priority alerts panel", value=True)
    max_rows    = st.slider("Max rows in ledger", 25, 500, 100, 25)

    st.markdown("---")
    st.caption("KYC Intelligence Platform · Hybrid Rule + ML Engine")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:24px;">
    <div>
        <p class="page-title">KYC Risk Intelligence</p>
        <p class="page-subtitle">Automated due diligence · Hybrid rule-based + Random Forest scoring</p>
    </div>
</div>
""", unsafe_allow_html=True)

if uploaded_file is None:
    st.markdown("""
    <div class="empty-state">
        <div style="font-size:56px;margin-bottom:24px;opacity:0.3">🔐</div>
        <h2>No data loaded</h2>
        <p>Upload <code>kyc_output.csv</code> in the sidebar to activate the dashboard.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Load & normalise ─────────────────────────────────────────────────────────
df = pd.read_csv(uploaded_file)
df.columns = df.columns.str.strip().str.lower()

# Ensure required columns exist with sensible fallbacks
for col in ['risk_score', 'rule_score', 'rf_score']:
    if col not in df.columns:
        df[col] = df.get('risk_score', 50)

df['decision_clean'] = df['decision'].apply(normalize_decision)
df['risk_tier_upper'] = df['risk_tier'].str.upper().str.strip()

total    = len(df)
approved = (df['decision_clean'] == 'APPROVE').sum()
manual   = (df['decision_clean'] == 'MANUAL REVIEW').sum()
rejected = (df['decision_clean'] == 'REJECT / EDD').sum()
avg_score = df['risk_score'].mean()
high_ct  = (df['risk_tier_upper'] == 'HIGH').sum()

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "  Executive Summary  ",
    "  Risk Analytics     ",
    "  Model Insights     ",
    "  Customer Ledger    ",
])

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    kpi_card("Total Evaluated", f"{total:,}", "customers in batch", "blue",  "👤", c1)
    kpi_card("Auto-Approved",   f"{approved:,}", f"{approved/total*100:.1f}% of batch", "green", "✅", c2)
    kpi_card("Manual Review",   f"{manual:,}",   f"{manual/total*100:.1f}% of batch",   "amber", "🔍", c3)
    kpi_card("Flagged / EDD",   f"{rejected:,}", f"{rejected/total*100:.1f}% of batch", "red",   "🚨", c4)
    kpi_card("Avg Risk Score",  f"{avg_score:.1f}", f"out of 100 · {high_ct} high-risk", "blue", "📊", c5)

    st.markdown("")

    # ── High-priority alerts ─────────────────────────────────────────────────
    if show_alerts:
        sanctions_ct = 0
        fraud_ct     = 0
        if 'sanctions_flag' in df.columns:
            sanctions_ct = (df['sanctions_flag'].astype(str).str.strip().isin(['1', '1.0', 'Yes', 'yes'])).sum()
        if 'fraud_history_flag' in df.columns:
            fraud_ct = (df['fraud_history_flag'].astype(str).str.strip().isin(['1', '1.0', 'Yes', 'yes'])).sum()

        if sanctions_ct > 0 or fraud_ct > 0:
            st.markdown('<p class="section-header">⚡ Priority Alerts</p>', unsafe_allow_html=True)
            a1, a2 = st.columns(2)
            if sanctions_ct > 0:
                a1.markdown(f"""
                <div class="alert-box critical">
                    <span style="font-size:20px">🛑</span>
                    <div>
                        <div style="font-weight:600;color:#991B1B;font-size:14px">{sanctions_ct} Sanctions Match{'es' if sanctions_ct>1 else ''}</div>
                        <div style="font-size:12px;color:#B45309;margin-top:2px">Immediate EDD required — blocked from auto-approval</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            if fraud_ct > 0:
                a2.markdown(f"""
                <div class="alert-box critical">
                    <span style="font-size:20px">⚠️</span>
                    <div>
                        <div style="font-weight:600;color:#991B1B;font-size:14px">{fraud_ct} Prior Fraud Histor{'ies' if fraud_ct>1 else 'y'}</div>
                        <div style="font-size:12px;color:#B45309;margin-top:2px">Manual review mandatory — high recidivism risk</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── Decision funnel ──────────────────────────────────────────────────────
    st.markdown('<p class="section-header">📋 Decision Breakdown</p>', unsafe_allow_html=True)

    funnel_col, pie_col = st.columns([3, 2])

    with funnel_col:
        funnel_data = {
            'Stage': ['Customers Evaluated', 'Auto-Approved', 'Manual Review', 'Flagged / EDD'],
            'Count': [total, approved, manual, rejected],
            'Color': ['#3B82F6', '#22C55E', '#F59E0B', '#EF4444'],
        }
        fig_funnel = go.Figure(go.Bar(
            x=funnel_data['Count'],
            y=funnel_data['Stage'],
            orientation='h',
            marker=dict(color=funnel_data['Color'], line=dict(width=0)),
            text=[f"{v:,} ({v/total*100:.1f}%)" for v in funnel_data['Count']],
            textposition='inside',
            textfont=dict(color='white', size=12, family='DM Sans'),
            hovertemplate='%{y}: %{x:,}<extra></extra>',
        ))
        fig_funnel.update_layout(
            **PLOTLY_BASE,
            height=220,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, tickfont=dict(size=13)),
            bargap=0.35,
        )
        st.plotly_chart(fig_funnel, use_container_width=True, config={'displayModeBar': False})

    with pie_col:
        tier_counts = df['risk_tier_upper'].value_counts().reset_index()
        tier_counts.columns = ['Tier', 'Count']
        fig_donut = go.Figure(go.Pie(
            labels=tier_counts['Tier'],
            values=tier_counts['Count'],
            hole=0.6,
            marker=dict(colors=[COLORS.get(t, '#94A3B8') for t in tier_counts['Tier']],
                        line=dict(color='white', width=3)),
            textfont=dict(size=12, family='DM Sans'),
            hovertemplate='%{label}: %{value} (%{percent})<extra></extra>',
        ))
        fig_donut.add_annotation(
            text=f"<b>{total}</b><br><span style='font-size:9px'>total</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=22, color='#0F172A', family='DM Sans'),
        )
        fig_donut.update_layout(**PLOTLY_BASE, height=220, showlegend=True,
                                 legend=dict(orientation='v', x=1.0, y=0.5, font=dict(size=12)))
        st.plotly_chart(fig_donut, use_container_width=True, config={'displayModeBar': False})

    # ── Risk factor frequency ────────────────────────────────────────────────
    if 'top_risk_factors' in df.columns:
        st.markdown('<p class="section-header">🎯 Top Risk Factors Triggered</p>', unsafe_allow_html=True)

        all_factors = (
            df['top_risk_factors']
            .dropna()
            .astype(str)
            .str.split(', ')
            .explode()
            .str.strip()
        )
        all_factors = all_factors[~all_factors.str.lower().isin(['none', '', 'nan'])]

        if not all_factors.empty:
            factor_counts = all_factors.value_counts().head(8).reset_index()
            factor_counts.columns = ['Factor', 'Count']
            factor_counts['Pct'] = (factor_counts['Count'] / total * 100).round(1)

            colors_gradient = ['#0F172A', '#1E293B', '#334155', '#475569',
                                '#64748B', '#94A3B8', '#CBD5E1', '#E2E8F0']

            fig_factors = go.Figure(go.Bar(
                y=factor_counts['Factor'],
                x=factor_counts['Count'],
                orientation='h',
                marker=dict(
                    color=colors_gradient[:len(factor_counts)],
                    line=dict(width=0),
                ),
                text=[f"{p}%" for p in factor_counts['Pct']],
                textposition='outside',
                textfont=dict(size=11, color='#64748B'),
                hovertemplate='%{y}<br>Count: %{x}<extra></extra>',
            ))
            fig_factors.update_layout(
                **PLOTLY_BASE,
                height=max(280, len(factor_counts) * 42),
                xaxis=dict(showgrid=True, gridcolor='#F1F5F9', zeroline=False),
                yaxis=dict(categoryorder='total ascending', tickfont=dict(size=12)),
                bargap=0.3,
            )
            st.plotly_chart(fig_factors, use_container_width=True, config={'displayModeBar': False})


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — RISK ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:

    # Score distribution histogram
    st.markdown('<p class="section-header">📈 Risk Score Distribution</p>', unsafe_allow_html=True)

    fig_hist = go.Figure()
    for tier, color in [('LOW', '#22C55E'), ('MEDIUM', '#F59E0B'), ('HIGH', '#EF4444')]:
        subset = df[df['risk_tier_upper'] == tier]['risk_score']
        if not subset.empty:
            fig_hist.add_trace(go.Histogram(
                x=subset, name=tier, marker_color=color,
                opacity=0.8, nbinsx=20,
                hovertemplate=f'{tier} Risk<br>Score: %{{x}}<br>Count: %{{y}}<extra></extra>',
            ))
    fig_hist.add_vline(x=40, line_dash='dash', line_color='#F59E0B',
                       annotation_text='Medium threshold (40)',
                       annotation_position='top right',
                       annotation_font_size=11)
    fig_hist.add_vline(x=75, line_dash='dash', line_color='#EF4444',
                       annotation_text='High threshold (75)',
                       annotation_position='top right',
                       annotation_font_size=11)
    fig_hist.update_layout(
        **PLOTLY_BASE,
        height=300,
        barmode='stack',
        xaxis=dict(title='Risk Score', showgrid=False),
        yaxis=dict(title='Customers', showgrid=True, gridcolor='#F1F5F9'),
        legend=dict(orientation='h', y=1.05, font=dict(size=12)),
    )
    st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False})

    # Score stats band
    q25, q50, q75 = df['risk_score'].quantile([0.25, 0.50, 0.75])
    s1, s2, s3, s4, s5 = st.columns(5)
    for col, label, val in [
        (s1, 'Minimum', df['risk_score'].min()),
        (s2, 'P25',     q25),
        (s3, 'Median',  q50),
        (s4, 'P75',     q75),
        (s5, 'Maximum', df['risk_score'].max()),
    ]:
        col.metric(label, f"{val:.1f}")

    # Account type breakdown (if available)
    if 'account_type' in df.columns:
        st.markdown('<p class="section-header">🏦 Risk by Account Type</p>', unsafe_allow_html=True)
        bar_data = (
            df.groupby(['account_type', 'risk_tier_upper'])
            .size()
            .reset_index(name='count')
        )
        fig_acct = px.bar(
            bar_data, x='account_type', y='count', color='risk_tier_upper',
            barmode='group', text_auto=True,
            color_discrete_map=COLORS,
            labels={'account_type': 'Account Type', 'count': 'Customers', 'risk_tier_upper': 'Risk Tier'},
        )
        fig_acct.update_traces(textfont_size=11)
        fig_acct.update_layout(**PLOTLY_BASE, height=320,
                                xaxis=dict(showgrid=False),
                                yaxis=dict(showgrid=True, gridcolor='#F1F5F9'),
                                legend=dict(orientation='h', y=1.05, font=dict(size=12)))
        st.plotly_chart(fig_acct, use_container_width=True, config={'displayModeBar': False})

    # Country risk heatmap (if available)
    if 'country_risk' in df.columns:
        st.markdown('<p class="section-header">🌍 Geography × Risk Tier</p>', unsafe_allow_html=True)
        geo_data = (
            df.groupby(['country_risk', 'risk_tier_upper'])
            .size()
            .reset_index(name='count')
        )
        fig_geo = px.bar(
            geo_data, x='country_risk', y='count', color='risk_tier_upper',
            barmode='stack', text_auto=True,
            color_discrete_map=COLORS,
            labels={'country_risk': 'Country Risk Level', 'count': 'Customers', 'risk_tier_upper': 'Tier'},
        )
        fig_geo.update_layout(**PLOTLY_BASE, height=280,
                               xaxis=dict(showgrid=False),
                               yaxis=dict(showgrid=True, gridcolor='#F1F5F9'),
                               legend=dict(orientation='h', y=1.05, font=dict(size=12)))
        st.plotly_chart(fig_geo, use_container_width=True, config={'displayModeBar': False})

    # Score vs transaction ratio scatter (if available)
    if 'txn_to_income_ratio' in df.columns:
        st.markdown('<p class="section-header">💹 Risk Score vs Transaction-to-Income Ratio</p>', unsafe_allow_html=True)
        fig_scatter = px.scatter(
            df, x='txn_to_income_ratio', y='risk_score',
            color='risk_tier_upper',
            color_discrete_map=COLORS,
            opacity=0.65,
            hover_data=['customer_id'] if 'customer_id' in df.columns else None,
            labels={'txn_to_income_ratio': 'Txn / Income Ratio', 'risk_score': 'Risk Score', 'risk_tier_upper': 'Tier'},
        )
        fig_scatter.update_traces(marker=dict(size=7, line=dict(width=0)))
        fig_scatter.update_layout(**PLOTLY_BASE, height=300,
                                   xaxis=dict(showgrid=True, gridcolor='#F1F5F9'),
                                   yaxis=dict(showgrid=True, gridcolor='#F1F5F9'),
                                   legend=dict(orientation='h', y=1.05, font=dict(size=12)))
        st.plotly_chart(fig_scatter, use_container_width=True, config={'displayModeBar': False})


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — MODEL INSIGHTS (RF vs Rule)
# ══════════════════════════════════════════════════════════════════════════════
with tab3:

    if not show_rf:
        st.info("Enable 'Show RF vs Rule Score breakdown' in the sidebar to view model insights.")
    else:
        st.markdown('<p class="section-header">🌲 Hybrid Scoring Architecture</p>', unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:12px;padding:20px 24px;margin-bottom:24px;">
            <div style="display:flex;gap:32px;flex-wrap:wrap;">
                <div>
                    <div style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:#94A3B8;margin-bottom:6px">Rule Engine Weight</div>
                    <div style="font-size:28px;font-weight:600;color:#0F172A">60%</div>
                    <div style="font-size:12px;color:#64748B">Transparent compliance logic</div>
                </div>
                <div>
                    <div style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:#94A3B8;margin-bottom:6px">Random Forest Weight</div>
                    <div style="font-size:28px;font-weight:600;color:#0F172A">40%</div>
                    <div style="font-size:12px;color:#64748B">Pattern detection & calibration</div>
                </div>
                <div>
                    <div style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:#94A3B8;margin-bottom:6px">Sanctions Override</div>
                    <div style="font-size:28px;font-weight:600;color:#EF4444">Hard</div>
                    <div style="font-size:12px;color:#64748B">Forces score → 100 always</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Rule vs RF scatter
        col_sc, col_diff = st.columns([3, 2])

        with col_sc:
            st.markdown("**Rule Score vs Random Forest Score**")
            fig_compare = go.Figure()
            fig_compare.add_trace(go.Scatter(
                x=df['rule_score'], y=df['rf_score'],
                mode='markers',
                marker=dict(
                    color=[COLORS.get(t, '#94A3B8') for t in df['risk_tier_upper']],
                    size=7, opacity=0.7, line=dict(width=0),
                ),
                hovertemplate='Rule: %{x:.1f}<br>RF: %{y:.1f}<extra></extra>',
            ))
            # Diagonal reference line
            max_val = max(df['rule_score'].max(), df['rf_score'].max())
            fig_compare.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode='lines',
                line=dict(color='#CBD5E1', dash='dot', width=1.5),
                hoverinfo='skip',
                showlegend=False,
            ))
            fig_compare.update_layout(
                **PLOTLY_BASE,
                height=320,
                xaxis=dict(title='Rule-Based Score', showgrid=True, gridcolor='#F1F5F9', range=[0, 105]),
                yaxis=dict(title='RF Score', showgrid=True, gridcolor='#F1F5F9', range=[0, 105]),
            )
            st.plotly_chart(fig_compare, use_container_width=True, config={'displayModeBar': False})

        with col_diff:
            st.markdown("**Score Delta Distribution (RF − Rule)**")
            df['score_delta'] = df['rf_score'] - df['rule_score']
            fig_delta = go.Figure(go.Histogram(
                x=df['score_delta'],
                nbinsx=20,
                marker_color='#3B82F6',
                opacity=0.75,
                hovertemplate='Delta: %{x:.0f}<br>Count: %{y}<extra></extra>',
            ))
            fig_delta.add_vline(x=0, line_dash='dash', line_color='#64748B')
            fig_delta.update_layout(
                **PLOTLY_BASE,
                height=320,
                xaxis=dict(title='RF − Rule', showgrid=False),
                yaxis=dict(title='Count', showgrid=True, gridcolor='#F1F5F9'),
            )
            st.plotly_chart(fig_delta, use_container_width=True, config={'displayModeBar': False})

        # Score averages per tier
        st.markdown('<p class="section-header">📊 Score Comparison by Risk Tier</p>', unsafe_allow_html=True)
        tier_avg = (
            df.groupby('risk_tier_upper')[['rule_score', 'rf_score', 'risk_score']]
            .mean()
            .round(1)
            .reset_index()
        )
        tier_avg.columns = ['Tier', 'Avg Rule Score', 'Avg RF Score', 'Avg Hybrid Score']
        fig_grouped = go.Figure()
        for col_name, color in [('Avg Rule Score', '#64748B'), ('Avg RF Score', '#3B82F6'), ('Avg Hybrid Score', '#0F172A')]:
            fig_grouped.add_trace(go.Bar(
                name=col_name,
                x=tier_avg['Tier'],
                y=tier_avg[col_name],
                marker=dict(color=color, line=dict(width=0)),
                text=tier_avg[col_name],
                textposition='outside',
                textfont=dict(size=11),
            ))
        fig_grouped.update_layout(
            **PLOTLY_BASE, height=280, barmode='group',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#F1F5F9', range=[0, 115]),
            legend=dict(orientation='h', y=1.05, font=dict(size=12)),
        )
        st.plotly_chart(fig_grouped, use_container_width=True, config={'displayModeBar': False})


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — CUSTOMER LEDGER
# ══════════════════════════════════════════════════════════════════════════════
with tab4:

    st.markdown('<p class="section-header">🔍 Search & Filter</p>', unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns([2, 1, 1, 1])
    with f1:
        search_id = st.text_input("Search customer ID or name", placeholder="Type to search…", label_visibility="collapsed")
    with f2:
        tier_filter = st.selectbox("Risk Tier", ["All Tiers", "HIGH", "MEDIUM", "LOW"], label_visibility="collapsed")
    with f3:
        decision_filter = st.selectbox("Decision", ["All Decisions", "APPROVE", "MANUAL REVIEW", "REJECT / EDD"], label_visibility="collapsed")
    with f4:
        score_min, score_max = st.select_slider(
            "Score range", options=list(range(0, 101, 5)),
            value=(0, 100), label_visibility="collapsed"
        )

    # Apply filters
    filtered = df.copy()
    if tier_filter != "All Tiers":
        filtered = filtered[filtered['risk_tier_upper'] == tier_filter]
    if decision_filter != "All Decisions":
        filtered = filtered[filtered['decision_clean'] == decision_filter]
    filtered = filtered[
        (filtered['risk_score'] >= score_min) & (filtered['risk_score'] <= score_max)
    ]
    if search_id and 'customer_id' in filtered.columns:
        filtered = filtered[filtered['customer_id'].astype(str).str.contains(search_id, case=False)]

    st.caption(f"Showing {min(max_rows, len(filtered)):,} of {len(filtered):,} matching customers")

    # ── Individual customer cards for high-risk ──────────────────────────────
    high_risk_filtered = filtered[filtered['risk_tier_upper'] == 'HIGH'].head(5)

    if not high_risk_filtered.empty and tier_filter in ['All Tiers', 'HIGH']:
        st.markdown('<p class="section-header">🚨 High-Risk Customer Profiles</p>', unsafe_allow_html=True)
        card_cols = st.columns(min(3, len(high_risk_filtered)))
        for idx, (_, row) in enumerate(high_risk_filtered.iterrows()):
            with card_cols[idx % len(card_cols)]:
                score = row['risk_score']
                cid   = row.get('customer_id', f'C{idx:04d}')
                tier  = row['risk_tier_upper']
                factors = str(row.get('top_risk_factors', 'Unknown')).split(', ')[:2]
                bar_color = score_color(score)

                st.markdown(f"""
                <div class="kpi-card red" style="margin-bottom:12px;">
                    <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px;">
                        <div>
                            <div class="kpi-label">{tier} RISK</div>
                            <div style="font-size:18px;font-weight:600;color:#0F172A;font-family:'DM Mono',monospace">{cid}</div>
                        </div>
                        <div style="font-size:26px;font-weight:700;color:{bar_color};font-family:'DM Mono',monospace">{score:.0f}</div>
                    </div>
                    <div class="score-bar-wrap">
                        <div class="score-bar-fill" style="width:{score}%;background:{bar_color};"></div>
                    </div>
                    <div style="margin-top:12px;">
                        {''.join(f'<div style="font-size:11px;color:#64748B;margin-top:4px">▸ {f.strip()}</div>' for f in factors if f.strip() and f.strip().lower() != 'nan')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── Full ledger table ────────────────────────────────────────────────────
    st.markdown('<p class="section-header">📋 Full Customer Ledger</p>', unsafe_allow_html=True)

    display_cols = ['customer_id', 'risk_score', 'risk_tier', 'decision', 'top_risk_factors']
    if show_rf and 'rule_score' in df.columns:
        display_cols = ['customer_id', 'risk_score', 'rule_score', 'rf_score', 'risk_tier', 'decision', 'top_risk_factors']

    # Keep only cols that exist
    display_cols = [c for c in display_cols if c in filtered.columns]

    display_df = filtered[display_cols].head(max_rows).copy()

    # Rename for display
    rename_map = {
        'customer_id':     'Customer ID',
        'risk_score':      'Hybrid Score',
        'rule_score':      'Rule Score',
        'rf_score':        'RF Score',
        'risk_tier':       'Tier',
        'decision':        'Decision',
        'top_risk_factors':'Top Risk Factors',
    }
    display_df = display_df.rename(columns={k: v for k, v in rename_map.items() if k in display_df.columns})

    # Round numeric columns
    for num_col in ['Hybrid Score', 'Rule Score', 'RF Score']:
        if num_col in display_df.columns:
            display_df[num_col] = display_df[num_col].round(1)

    def style_row(row):
        styles = [''] * len(row)
        if 'Decision' in row.index:
            dec = str(row['Decision']).upper()
            if 'REJECT' in dec or 'EDD' in dec:
                styles[row.index.get_loc('Decision')] = 'background-color:#FEF2F2;color:#991B1B;font-weight:600'
            elif 'MANUAL' in dec:
                styles[row.index.get_loc('Decision')] = 'background-color:#FEF3C7;color:#92400E;font-weight:600'
            elif dec == 'APPROVE':
                styles[row.index.get_loc('Decision')] = 'background-color:#DCFCE7;color:#166534;font-weight:600'
        if 'Tier' in row.index:
            t = str(row['Tier']).upper()
            if t == 'HIGH':
                styles[row.index.get_loc('Tier')] = 'background-color:#FEE2E2;color:#991B1B;font-weight:600'
            elif t == 'MEDIUM':
                styles[row.index.get_loc('Tier')] = 'background-color:#FEF3C7;color:#92400E;font-weight:600'
            else:
                styles[row.index.get_loc('Tier')] = 'background-color:#DCFCE7;color:#166534;font-weight:600'
        return styles

    st.dataframe(
        display_df.style.apply(style_row, axis=1),
        use_container_width=True,
        hide_index=True,
        height=min(600, 40 + len(display_df) * 35),
    )

    # ── CSV download ─────────────────────────────────────────────────────────
    csv_data = filtered.to_csv(index=False).encode()
    st.download_button(
        label="⬇ Download Filtered Results as CSV",
        data=csv_data,
        file_name='kyc_filtered_results.csv',
        mime='text/csv',
        use_container_width=True,
    )
