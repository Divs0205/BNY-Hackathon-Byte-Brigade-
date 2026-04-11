import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

print("🚀 Starting KYC Risk Engine...")

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
try:
    df = pd.read_csv('raw_data.csv')
    print(f"✅ Loaded {len(df)} records.")
except FileNotFoundError:
    print("❌ ERROR: Could not find 'raw_data.csv'. Make sure the file is in the same folder!")
    exit()

# Clean column names
df.columns = df.columns.str.strip().str.lower()

# ─────────────────────────────────────────────
# 2. PREPROCESSING & FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("⚙️ Processing features...")

# Fill missing numbers with median, text with 'Unknown'
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
df.fillna('Unknown', inplace=True)

# Map text to numbers
df['country_risk_encoded'] = (
    df['country_risk']
    .map({'Low': 1, 'Medium': 2, 'High': 3, 'Unknown': 2})
    .fillna(2)
)

df['doc_status_encoded'] = (
    df['document_status']
    .map({'Valid': 1, 'Complete': 1, 'Missing': 0, 'Expired': 0, 'Partial': 0, 'Unknown': 0})
    .fillna(0)
)

# Binary flags — handle Yes/No strings and numeric values
for col in ['pep_flag', 'sanctions_flag', 'adverse_media_flag', 'fraud_history_flag', 'address_verified']:
    if col in df.columns:
        df[col] = (
            df[col]
            .replace({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0})
            .pipe(pd.to_numeric, errors='coerce')
            .fillna(0)
            .astype(int)
        )
    else:
        df[col] = 0  # Fallback if column is missing

# Derived / power features
df['composite_compliance_risk'] = (
    df['pep_flag'] + df['sanctions_flag'] + df['adverse_media_flag']
)

if 'annual_income' in df.columns and 'monthly_txn_count' in df.columns:
    df['txn_to_income_ratio'] = df['monthly_txn_count'] / (df['annual_income'] + 1)
else:
    df['txn_to_income_ratio'] = 0

tenure_col = df['customer_tenure_years'] if 'customer_tenure_years' in df.columns else pd.Series(1, index=df.index)
df['is_new_customer'] = (pd.to_numeric(tenure_col, errors='coerce').fillna(1) == 0).astype(int)

# ─────────────────────────────────────────────
# 3. SCORING & EXPLAINABILITY ENGINE
# ─────────────────────────────────────────────
print("🧠 Calculating Risk Scores and Explanations...")

def evaluate_customer(row):
    score = 0
    factors = []

    # CRITICAL — hard stops
    if row.get('sanctions_flag', 0) == 1:
        score += 100
        factors.append("Sanctions List Match")
    if row.get('fraud_history_flag', 0) == 1:
        score += 80
        factors.append("Prior Fraud History")

    # HIGH impact
    if row.get('pep_flag', 0) == 1:
        score += 40
        factors.append("Politically Exposed Person (PEP)")
    if row.get('adverse_media_flag', 0) == 1:
        score += 40
        factors.append("Adverse Media Found")
    if row.get('doc_status_encoded', 1) == 0:
        score += 35
        factors.append("Missing/Invalid KYC Docs")

    # MEDIUM impact
    if row.get('country_risk_encoded', 1) == 3:
        score += 25
        factors.append("High-Risk Geography")
    if row.get('digital_risk_score', 0) > 75:
        score += 20
        factors.append("High Digital Device Risk")
    if row.get('address_verified', 1) == 0:
        score += 15
        factors.append("Unverified Address")

    # LOW impact
    if row.get('is_new_customer', 0) == 1:
        score += 10
    if row.get('txn_to_income_ratio', 0) > 0.05:
        score += 10

    final_score = min(score, 100)
    top_factors = ", ".join(factors[:3]) if factors else "None"

    # Decision mapping
    if final_score >= 75 or row.get('sanctions_flag', 0) == 1:
        tier, decision = 'HIGH', 'REJECT/EDD'
    elif final_score >= 40:
        tier, decision = 'MEDIUM', 'MANUAL_REVIEW'
    else:
        tier, decision = 'LOW', 'APPROVE'

    return pd.Series([final_score, tier, decision, top_factors])


df[['risk_score', 'risk_tier', 'decision', 'top_risk_factors']] = df.apply(evaluate_customer, axis=1)

# ─────────────────────────────────────────────
# 4. EXPORT RESULTS
# ─────────────────────────────────────────────
print("💾 Saving final kyc_output.csv...")

# Auto-generate customer_id if not present
if 'customer_id' not in df.columns:
    df['customer_id'] = ['C' + str(i).zfill(4) for i in range(1, len(df) + 1)]

final_columns = ['customer_id', 'risk_score', 'risk_tier', 'decision', 'top_risk_factors']
df[final_columns].to_csv('kyc_output.csv', index=False)

print("✅ SUCCESS! 'kyc_output.csv' has been generated. Upload it to your Streamlit dashboard now!")

# ─────────────────────────────────────────────
# 5. QUICK SUMMARY PRINT
# ─────────────────────────────────────────────
total    = len(df)
approved = (df['decision'] == 'APPROVE').sum()
manual   = (df['decision'] == 'MANUAL_REVIEW').sum()
flagged  = df['decision'].str.contains('REJECT|EDD', na=False).sum()

print(f"\n{'═' * 45}")
print(f"  BATCH SUMMARY")
print(f"{'═' * 45}")
print(f"  Total evaluated  : {total}")
print(f"  ✅ Approved       : {approved} ({approved/total*100:.1f}%)")
print(f"  🟡 Manual Review  : {manual}   ({manual/total*100:.1f}%)")
print(f"  🔴 Flagged/EDD    : {flagged}  ({flagged/total*100:.1f}%)")
print(f"  Avg risk score   : {df['risk_score'].mean():.1f}")
print(f"{'═' * 45}\n")
