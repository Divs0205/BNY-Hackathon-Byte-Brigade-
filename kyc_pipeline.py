import pandas as pd
import numpy as np
import pickle
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Ensure model folder exists
os.makedirs("model", exist_ok=True)


# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────
def preprocess_data(df):
    df = df.copy()

    # Fill missing values
    df.fillna(0, inplace=True)

    # Encode categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    return df


# ─────────────────────────────────────────────
# TRAIN MODEL
# ─────────────────────────────────────────────
def train_model(df):
    df = preprocess_data(df)

    if "risk_label" not in df.columns:
        raise ValueError("Dataset must contain 'risk_label' column")

    X = df.drop("risk_label", axis=1)
    y = df["risk_label"]

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42
    )

    model.fit(X, y)

    # Save model
    with open("model/model.pkl", "wb") as f:
        pickle.dump(model, f)

    return "Model trained successfully!"


# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────
def predict(df):
    df = preprocess_data(df)

    # Load model
    model_path = "model/model.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Train it first.")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    predictions = model.predict(df)

    return predictions


# ─────────────────────────────────────────────
# OPTIONAL: RISK SCORE (0–100)
# ─────────────────────────────────────────────
def compute_risk_score(df):
    df = preprocess_data(df)

    model_path = "model/model.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Train it first.")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    probs = model.predict_proba(df)[:, 1]  # probability of high risk

    scores = (probs * 100).astype(int)

    return scores
