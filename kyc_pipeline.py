"""
Smart KYC Risk Scoring Engine
==============================
Rule-based weighted scoring system for customer risk assessment.
Score range: 0–100 (higher = riskier)
"""

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# SCORING WEIGHTS & MAPPINGS (easy to modify)
# ─────────────────────────────────────────────

WEIGHTS = {
    "sanctions_flag":      30,   # Critical
    "fraud_history_flag":  20,   # Critical
    "pep_flag":            10,   # High
    "adverse_media_flag":   8,   # High
    "document_status":      7,   # High
    "country_risk":         6,   # Medium
    "address_verified":     5,   # Medium
    "digital_risk_score":   5,   # Medium  (scaled 0–5)
    "age":                  3,   # Low
    "customer_tenure_years":3,   # Low
    "txn_behavior":         3,   # Low (derived)
}
# Total max weight = 100

COUNTRY_RISK_MAP = {
    "low":    0.0,
    "medium": 0.5,
    "high":   1.0,
}

DOCUMENT_STATUS_MAP = {
    "complete":   0.0,
    "pending":    0.5,
    "incomplete": 1.0,
    "expired":    1.0,
    "missing":    1.0,
}

OCCUPATION_RISK_MAP = {
    # Higher-risk occupations get a small additive bump (0–5 pts)
    "politician":       5,
    "government":       4,
    "casino":           4,
    "money services":   4,
    "real estate":      3,
    "legal":            2,
    "finance":          2,
    "other":            1,
    "salaried":         0,
    "retired":          0,
    "student":          0,
}

ACCOUNT_TYPE_RISK_MAP = {
    # Some account types carry higher risk
    "offshore":     5,
    "business":     3,
    "joint":        2,
    "savings":      1,
    "current":      1,
    "student":      0,
}


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def _safe_flag(series: pd.Series) -> pd.Series:
    """Coerce a binary flag column to 0/1, defaulting missing to 0."""
    return pd.to_numeric(series, errors="coerce").fillna(0).clip(0, 1)


def _map_category(series: pd.Series, mapping: dict, default: float = 0.5) -> pd.Series:
    """Map a string column via a dict; unknown/missing values get `default`."""
    return (
        series.astype(str)
              .str.strip()
              .str.lower()
              .map(mapping)
              .fillna(default)
    )


def _age_risk(age_series: pd.Series) -> pd.Series:
    """
    Age risk (0–1):
      - Very young (<21) or very old (>75) → higher risk (less stable)
      - Prime working age (25–60) → lowest risk
    """
    age = pd.to_numeric(age_series, errors="coerce").fillna(35)
    risk = pd.Series(0.3, index=age.index)           # default moderate
    risk = risk.where(age >= 21, 0.7)                # <21 → higher
    risk = risk.where(age <= 75, 0.6)                # >75 → higher
    risk[(age >= 25) & (age <= 60)] = 0.0            # prime → low
    return risk


def _tenure_risk(tenure_series: pd.Series) -> pd.Series:
    """
    Tenure risk (0–1):
      - New customers (<1 yr) → high risk
      - 1–3 yrs → medium
      - >3 yrs → low
    """
    tenure = pd.to_numeric(tenure_series, errors="coerce").fillna(0).clip(lower=0)
    risk = pd.Series(0.0, index=tenure.index)
    risk[tenure < 1]               = 1.0
    risk[(tenure >= 1) & (tenure < 3)] = 0.5
    # tenure >= 3 stays 0.0
    return risk


def _txn_behavior_risk(txn_series: pd.Series, income_series: pd.Series) -> pd.Series:
    """
    Transaction behaviour risk (0–1):
      - High txn count relative to income → more suspicious
      - >100 txns/month → high risk
      - 30–100 → medium
      - <30 → low
    Also penalises very low income with high txn count.
    """
    txn    = pd.to_numeric(txn_series,    errors="coerce").fillna(0).clip(lower=0)
    income = pd.to_numeric(income_series, errors="coerce").fillna(1).clip(lower=1)

    risk = pd.Series(0.0, index=txn.index)
    risk[txn > 100] = 1.0
    risk[(txn >= 30) & (txn <= 100)] = 0.5

    # Extra penalty: high txn count but low income (< 20k annual)
    high_txn_low_income = (txn > 50) & (income < 20_000)
    risk[high_txn_low_income] = risk[high_txn_low_income].clip(lower=0.8)

    return risk


def _digital_risk_scaled(digital_series: pd.Series) -> pd.Series:
    """Rescale digital_risk_score (0–100) → (0–1)."""
    score = pd.to_numeric(digital_series, errors="coerce").fillna(50).clip(0, 100)
    return score / 100.0


def _occupation_bonus(occupation_series: pd.Series) -> pd.Series:
    """Return additive bonus points (0–5) based on occupation risk."""
    return (
        occupation_series.astype(str)
                         .str.strip()
                         .str.lower()
                         .map(OCCUPATION_RISK_MAP)
                         .fillna(1)   # unknown occupation → small penalty
    )


def _account_type_bonus(account_series: pd.Series) -> pd.Series:
    """Return additive bonus points (0–5) based on account type risk."""
    return (
        account_series.astype(str)
                      .str.strip()
                      .str.lower()
                      .map(ACCOUNT_TYPE_RISK_MAP)
                      .fillna(1)
    )


# ─────────────────────────────────────────────
# MAIN SCORING FUNCTION
# ─────────────────────────────────────────────

def compute_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a KYC risk score (0–100) for each row in df.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the columns listed in DATASET FEATURES.

    Returns
    -------
    pd.DataFrame
        Original df with an added 'risk_score' column (int, 0–100).
    """
    df = df.copy()   # never mutate caller's DataFrame

    # ── 1. Critical flags ──────────────────────────────────────────────
    sanctions    = _safe_flag(df["sanctions_flag"])      # 0 or 1
    fraud        = _safe_flag(df["fraud_history_flag"])  # 0 or 1

    score  = sanctions * WEIGHTS["sanctions_flag"]       # 0 or 30
    score += fraud     * WEIGHTS["fraud_history_flag"]   # 0 or 20

    # ── 2. High-impact flags ───────────────────────────────────────────
    pep          = _safe_flag(df["pep_flag"])
    adverse      = _safe_flag(df["adverse_media_flag"])
    doc_risk     = _map_category(df["document_status"], DOCUMENT_STATUS_MAP, default=0.5)

    score += pep     * WEIGHTS["pep_flag"]               # 0 or 10
    score += adverse * WEIGHTS["adverse_media_flag"]     # 0 or 8
    score += doc_risk * WEIGHTS["document_status"]       # 0–7

    # ── 3. Medium-impact features ──────────────────────────────────────
    country_risk = _map_category(df["country_risk"], COUNTRY_RISK_MAP, default=0.5)
    addr_ok      = _safe_flag(df["address_verified"])
    addr_risk    = 1.0 - addr_ok                         # unverified = risky
    dig_risk     = _digital_risk_scaled(df["digital_risk_score"])

    score += country_risk * WEIGHTS["country_risk"]      # 0–6
    score += addr_risk    * WEIGHTS["address_verified"]  # 0 or 5
    score += dig_risk     * WEIGHTS["digital_risk_score"]# 0–5

    # ── 4. Low-impact behavioural features ────────────────────────────
    age_r    = _age_risk(df["age"])
    tenure_r = _tenure_risk(df["customer_tenure_years"])
    txn_r    = _txn_behavior_risk(df["monthly_txn_count"], df["annual_income"])

    score += age_r    * WEIGHTS["age"]                   # 0–3
    score += tenure_r * WEIGHTS["customer_tenure_years"] # 0–3
    score += txn_r    * WEIGHTS["txn_behavior"]          # 0–3

    # ── 5. Additive bonuses (occupation + account type) ───────────────
    score += _occupation_bonus(df["occupation"])         # 0–5
    score += _account_type_bonus(df["account_type"])     # 0–5

    # ── 6. Hard floor rules ────────────────────────────────────────────
    # Sanctioned customer → minimum score of 90
    score = score.where(sanctions == 0, score.clip(lower=90))

    # Fraud history → minimum score of 70
    score = score.where(fraud == 0, score.clip(lower=70))

    # ── 7. Clamp to [0, 100] and round ────────────────────────────────
    df["risk_score"] = score.clip(0, 100).round().astype(int)

    return df


# ─────────────────────────────────────────────
# SAMPLE DATASET
# ─────────────────────────────────────────────

sample_data = {
    "sanctions_flag":        [1,  0,  0,  0,  0,  0,  1,  0,  0,  0],
    "fraud_history_flag":    [0,  1,  0,  0,  0,  0,  0,  1,  0,  0],
    "pep_flag":              [0,  0,  1,  0,  0,  0,  1,  0,  0,  0],
    "adverse_media_flag":    [0,  1,  0,  1,  0,  0,  0,  0,  0,  0],
    "document_status":       ["complete","incomplete","complete","pending",
                              "complete","expired","incomplete","complete",
                              "complete","missing"],
    "address_verified":      [1,  0,  1,  0,  1,  1,  0,  1,  1,  0],
    "country_risk":          ["low","high","medium","high","low","medium",
                              "high","low","low","medium"],
    "digital_risk_score":    [20, 85,  40,  70,  15,  60,  90,  35,  10,  55],
    "customer_tenure_years": [5,  0.5, 3,   1,   8,   2,   0.2, 4,   10,  0.1],
    "monthly_txn_count":     [15, 120, 40,  80,  10,  55,  200, 30,  8,   95],
    "annual_income":         [60000, 15000, 90000, 25000, 120000,
                              40000, 8000,  55000, 200000, 18000],
    "age":                   [35, 28,  52,  19,  45,  70,  33,  60,  41,  22],
    "occupation":            ["salaried","money services","finance","student",
                              "retired","real estate","politician","salaried",
                              "legal","casino"],
    "account_type":          ["current","business","savings","student",
                              "savings","joint","offshore","current",
                              "current","business"],
}

df_sample = pd.DataFrame(sample_data)


# ─────────────────────────────────────────────
# RUN & PRINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    result = compute_score(df_sample)

    # Pretty-print selected columns + score
    display_cols = [
        "sanctions_flag", "fraud_history_flag", "pep_flag",
        "adverse_media_flag", "document_status", "country_risk",
        "digital_risk_score", "occupation", "account_type",
        "risk_score",
    ]

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)

    print("=" * 80)
    print("  SMART KYC RISK SCORING ENGINE — RESULTS")
    print("=" * 80)
    print(result[display_cols].to_string(index=True))
    print()

    # Risk band summary
    def risk_band(score):
        if score >= 80: return "🔴 CRITICAL"
        if score >= 60: return "🟠 HIGH"
        if score >= 40: return "🟡 MEDIUM"
        return              "🟢 LOW"

    result["risk_band"] = result["risk_score"].apply(risk_band)
    print("=" * 80)
    print("  RISK BAND SUMMARY")
    print("=" * 80)
    print(result[["risk_score", "risk_band"]].to_string(index=True))
    print()
    print(f"  Average risk score : {result['risk_score'].mean():.1f}")
    print(f"  Max risk score     : {result['risk_score'].max()}")
    print(f"  Min risk score     : {result['risk_score'].min()}")
    print("=" * 80)
