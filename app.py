import streamlit as st
import pandas as pd
import plotly.express as px

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="BNY KYC Engine", page_icon="🏦", layout="wide")

st.markdown("""
    <style>
    .stMetric {background-color: #ffffff; padding: 15px; border-radius: 8px; border: 1px solid #e0e0e0;}
    .stTabs [data-baseweb="tab-list"] {gap: 24px;}
    .stTabs [data-baseweb="tab"] {height: 50px; white-space: pre-wrap; background-color: transparent;
        border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px;}
    </style>
""", unsafe_allow_html=True)

# --- 2. SIDEBAR ---
st.sidebar.title("⚙️ Engine Controls")
st.sidebar.markdown("Upload the batch processing output from the KYC Engine.")
uploaded_file = st.sidebar.file_uploader("Upload kyc_output.csv", type=["csv"])

# --- 3. MAIN HEADER ---
st.title("🏦 Smart KYC Risk Scoring Engine")
st.markdown("Automated Due Diligence & Customer Risk Classification Dashboard")

# --- 4. MAIN LOGIC ---
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.lower()

    tab1, tab2 = st.tabs(["📊 Executive Summary", "🗄️ Searchable Ledger"])

    # ── TAB 1: Executive Summary ──────────────────────────────────
    with tab1:
        # KPI Metrics
        total    = len(df)
        approved = len(df[df['decision'].str.upper() == 'APPROVE'])
        manual   = len(df[df['decision'].str.upper() == 'MANUAL_REVIEW'])
        rejected = len(df[df['decision'].str.upper().isin(['REJECT', 'EDD', 'REJECT/EDD'])])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Customers Evaluated", total)
        col2.metric("🟢 Auto-Approved",  f"{approved} ({(approved / total) * 100:.1f}%)")
        col3.metric("🟡 Manual Review",  f"{manual}   ({(manual   / total) * 100:.1f}%)")
        col4.metric("🔴 Flagged / EDD",  f"{rejected} ({(rejected / total) * 100:.1f}%)")

        st.divider()

        # --- Visual Analytics ---
        st.subheader("Visual Analytics")

        # Top row: Pie + Bar
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("**Distribution by Risk Tier**")
            tier_counts = df['risk_tier'].value_counts().reset_index()
            tier_counts.columns = ['Risk Tier', 'Count']
            fig_pie = px.pie(
                tier_counts, names='Risk Tier', values='Count',
                color='Risk Tier',
                color_discrete_map={
                    'LOW': '#28a745', 'MEDIUM': '#ffc107', 'HIGH': '#dc3545',
                    'low': '#28a745', 'medium': '#ffc107', 'high': '#dc3545',
                },
                hole=0.45,
            )
            fig_pie.update_layout(margin=dict(t=20, b=20, l=0, r=0), height=300)
            st.plotly_chart(fig_pie, use_container_width=True)

        with chart_col2:
            st.markdown("**Top AML Risk Factors Triggered**")
            if 'top_risk_factors' in df.columns:
                all_factors = df['top_risk_factors'].dropna().astype(str).str.split(', ').sum()
                all_factors = [f for f in all_factors if f.strip().lower() not in ('none', '')]
                if all_factors:
                    factor_counts = pd.Series(all_factors).value_counts().head(5).reset_index()
                    factor_counts.columns = ['Risk Factor', 'Frequency']
                    fig_bar = px.bar(
                        factor_counts, x='Frequency', y='Risk Factor', orientation='h',
                        color='Frequency', color_continuous_scale='Reds',
                    )
                    fig_bar.update_layout(
                        yaxis={'categoryorder': 'total ascending'},
                        margin=dict(t=20, b=20, l=0, r=0), height=300,
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.info("No major risk factors detected in this batch.")
            else:
                st.warning("'top_risk_factors' column not found in uploaded file.")

        st.divider()

        # Bottom row: Risk Profiles by Account Type
        st.markdown("**Business Insight: Risk Profiles by Account Type**")
        if 'account_type' in df.columns and 'risk_tier' in df.columns:
            bar_data = df.groupby(['account_type', 'risk_tier']).size().reset_index(name='count')
            fig_stacked = px.bar(
                bar_data, x='account_type', y='count', color='risk_tier',
                color_discrete_map={
                    'LOW': '#28a745', 'MEDIUM': '#ffc107', 'HIGH': '#dc3545',
                    'low': '#28a745', 'medium': '#ffc107', 'high': '#dc3545',
                },
                barmode='group',
                text_auto=True,
            )
            fig_stacked.update_layout(
                margin=dict(t=20, b=20, l=0, r=0),
                xaxis_title="Account Type",
                yaxis_title="Number of Customers",
                legend_title="Risk Tier",
            )
            st.plotly_chart(fig_stacked, use_container_width=True)
        else:
            st.warning("Account Type data not available for this view.")

    # ── TAB 2: Searchable Ledger ──────────────────────────────────
    with tab2:
        st.subheader("Individual Customer Decisions")

        f_col1, f_col2 = st.columns([1, 2])
        with f_col1:
            tier_filter = st.selectbox("Filter by Risk Tier", ["All", "LOW", "MEDIUM", "HIGH"])
        with f_col2:
            search_id = st.text_input("🔍 Search by Customer ID", placeholder="e.g. C001")

        filtered_df = df.copy()
        if tier_filter != "All":
            filtered_df = filtered_df[filtered_df['risk_tier'].str.upper() == tier_filter]
        if search_id and 'customer_id' in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df['customer_id'].astype(str).str.contains(search_id, case=False)
            ]

        def color_decisions(val):
            v = str(val).upper()
            if v == 'APPROVE':
                return 'background-color:#e8f5e9;color:#2e7d32;font-weight:bold'
            if v in ('REJECT', 'EDD', 'REJECT/EDD'):
                return 'background-color:#ffebee;color:#c62828;font-weight:bold'
            if v == 'MANUAL_REVIEW':
                return 'background-color:#fff8e1;color:#f57f17;font-weight:bold'
            return ''

        st.dataframe(
            filtered_df.style.map(color_decisions, subset=['decision']),
            use_container_width=True,
            hide_index=True,
        )

        # Optional risk reason column
        if 'international_txn' in filtered_df.columns and 'num_transactions' in filtered_df.columns:
            def risk_reason(row):
                if row['international_txn'] == 1:
                    return 'International transactions detected'
                if row['num_transactions'] > 50:
                    return 'High transaction frequency'
                return 'Normal activity'

            filtered_df['risk_reason'] = filtered_df.apply(risk_reason, axis=1)
            st.dataframe(filtered_df[['customer_id', 'risk_reason']], use_container_width=True, hide_index=True)

# --- 5. EMPTY STATE ---
else:
    st.info("👈 Please upload 'kyc_output.csv' in the sidebar to populate the dashboard.")
    st.image(
        "https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&q=80&w=1000",
        caption="Awaiting Data Ingestion...",
        use_container_width=True,
    )
