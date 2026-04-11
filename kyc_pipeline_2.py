import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

# ============================
# YOUR EXISTING SCORING LOGIC
# ============================




# ============================
# TRAIN MODEL FUNCTION
# ============================

def train_model(df):
    """
    Train ML model using computed risk score as target
    """

    # Step 1: Compute risk score
    df = compute_score(df)

    # Step 2: Create label (binary classification)
    # High risk >= 60 → 1, else 0
    df["risk_label"] = (df["risk_score"] >= 60).astype(int)

    # Step 3: Select features
    features = [
        "sanctions_flag",
        "fraud_history_flag",
        "pep_flag",
        "adverse_media_flag",
        "address_verified",
        "digital_risk_score",
        "customer_tenure_years",
        "monthly_txn_count",
        "annual_income",
        "age"
    ]

    X = df[features]
    y = df["risk_label"]

    # Step 4: Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Step 5: Save model
    joblib.dump(model, "model/model.pkl")

    return model


# ============================
# PREDICT FUNCTION
# ============================

def predict(df):
    """
    Load model and predict risk
    """

    # Step 1: Compute score first
    df = compute_score(df)

    # Step 2: Load model
    model = joblib.load("model/model.pkl")

    features = [
        "sanctions_flag",
        "fraud_history_flag",
        "pep_flag",
        "adverse_media_flag",
        "address_verified",
        "digital_risk_score",
        "customer_tenure_years",
        "monthly_txn_count",
        "annual_income",
        "age"
    ]

    X = df[features]

    # Step 3: Predict
    df["ml_prediction"] = model.predict(X)

    return df
