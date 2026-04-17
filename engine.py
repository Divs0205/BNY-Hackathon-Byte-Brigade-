import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
import numpy as np

print("🚀 Starting KYC Risk Engine (with Random Forest)...")

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
# 3. RULE-BASED SCORING & EXPLAINABILITY ENGINE
# ─────────────────────────────────────────────
print("🧠 Calculating Rule-Based Risk Scores and Explanations...")

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


df[['rule_score', 'risk_tier_rule', 'decision_rule', 'top_risk_factors']] = df.apply(evaluate_customer, axis=1)

# ─────────────────────────────────────────────
# 4. RANDOM FOREST MODEL
# ─────────────────────────────────────────────
print("🌲 Training Random Forest model...")

# Features used by the RF model
RF_FEATURES = [
    'pep_flag',
    'sanctions_flag',
    'adverse_media_flag',
    'fraud_history_flag',
    'address_verified',
    'country_risk_encoded',
    'doc_status_encoded',
    'composite_compliance_risk',
    'txn_to_income_ratio',
    'is_new_customer',
]

# Add digital_risk_score if available
if 'digital_risk_score' in df.columns:
    RF_FEATURES.append('digital_risk_score')

# Build feature matrix — all columns guaranteed to exist after preprocessing
X = df[RF_FEATURES].copy()

# Derive labels from rule-based scores for supervised training
# 0 = LOW, 1 = MEDIUM, 2 = HIGH
def score_to_label(score):
    if score >= 75:
        return 2
    elif score >= 40:
        return 1
    else:
        return 0

y = df['rule_score'].apply(score_to_label)

# Train/test split (stratified to preserve class balance)
if len(df) >= 10:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
else:
    # Too few records — train on everything
    X_train, X_test, y_train, y_test = X, X, y, y

# Build and calibrate the Random Forest
rf_base = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=2,
    class_weight='balanced',   # handles imbalanced HIGH/LOW classes
    random_state=42,
    n_jobs=-1
)

# Calibrate for reliable probability estimates (Platt scaling)
rf_model = CalibratedClassifierCV(rf_base, cv='prefit' if len(df) < 10 else 5, method='sigmoid')

if len(df) >= 10:
    rf_base.fit(X_train, y_train)
    rf_model = CalibratedClassifierCV(rf_base, cv=min(5, len(X_train)), method='sigmoid')
    rf_model.fit(X_train, y_train)
else:
    rf_base.fit(X_train, y_train)
    rf_model = rf_base   # skip calibration for tiny datasets

# Predict class probabilities on all records
rf_probs = rf_model.predict_proba(X)  # shape: (n, 3) — LOW / MEDIUM / HIGH

# Convert RF HIGH-risk probability to a 0–100 score
# Classes are ordered by the classifier: check actual order
classes = list(rf_model.classes_) if hasattr(rf_model, 'classes_') else [0, 1, 2]
high_idx  = classes.index(2) if 2 in classes else -1
med_idx   = classes.index(1) if 1 in classes else -1

if high_idx >= 0 and med_idx >= 0:
    # Weighted: HIGH prob * 100 + MEDIUM prob * 50
    df['rf_score'] = (rf_probs[:, high_idx] * 100 + rf_probs[:, med_idx] * 50).clip(0, 100)
elif high_idx >= 0:
    df['rf_score'] = (rf_probs[:, high_idx] * 100).clip(0, 100)
else:
    df['rf_score'] = df['rule_score']   # safe fallback

print(f"   RF model trained on {len(X_train)} records | classes: {classes}")

# Feature importances (informational)
try:
    importances = rf_base.feature_importances_
    fi_df = pd.DataFrame({'feature': RF_FEATURES, 'importance': importances})
    fi_df = fi_df.sort_values('importance', ascending=False)
    print("\n   📊 Top Feature Importances (Random Forest):")
    for _, row in fi_df.head(5).iterrows():
        print(f"      {row['feature']:35s}  {row['importance']:.4f}")
    print()
except Exception:
    pass

# ─────────────────────────────────────────────
# 5. HYBRID SCORE (Rule-based + RF Blend)
# ─────────────────────────────────────────────
# Blend weights: 60% rule-based (transparent compliance logic) +
#                40% Random Forest (pattern detection)
RULE_WEIGHT = 0.60
RF_WEIGHT   = 0.40

df['risk_score'] = (
    RULE_WEIGHT * df['rule_score'] + RF_WEIGHT * df['rf_score']
).clip(0, 100).round(1)

# Hard override: sanctions always → HIGH regardless of blend
sanctions_mask = df['sanctions_flag'] == 1
df.loc[sanctions_mask, 'risk_score'] = 100.0

# Final tier and decision from hybrid score
def assign_tier_decision(row):
    score = row['risk_score']
    if score >= 75 or row['sanctions_flag'] == 1:
        return pd.Series(['HIGH', 'REJECT/EDD'])
    elif score >= 40:
        return pd.Series(['MEDIUM', 'MANUAL_REVIEW'])
    else:
        return pd.Series(['LOW', 'APPROVE'])

df[['risk_tier', 'decision']] = df.apply(assign_tier_decision, axis=1)

# ─────────────────────────────────────────────
# 6. EXPORT RESULTS
# ─────────────────────────────────────────────
print("💾 Saving final kyc_output.csv...")

# Auto-generate customer_id if not present
if 'customer_id' not in df.columns:
    df['customer_id'] = ['C' + str(i).zfill(4) for i in range(1, len(df) + 1)]

final_columns = [
    'customer_id',
    'risk_score',       # hybrid score (rule + RF blend)
    'rule_score',       # rule-based score (for audit/explainability)
    'rf_score',         # random forest score (for audit/explainability)
    'risk_tier',
    'decision',
    'top_risk_factors'  # from rule-based engine (human-readable)
]
df[final_columns].to_csv('kyc_output.csv', index=False)

print("✅ SUCCESS! 'kyc_output.csv' has been generated. Upload it to your Streamlit dashboard now!")

# ─────────────────────────────────────────────
# 7. QUICK SUMMARY PRINT
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
print(f"  Avg hybrid score : {df['risk_score'].mean():.1f}")
print(f"  Avg rule score   : {df['rule_score'].mean():.1f}")
print(f"  Avg RF score     : {df['rf_score'].mean():.1f}")
print(f"{'═' * 45}\n")
