import streamlit as st
import pandas as pd
from kyc_pipeline import train_model, predict

st.title("🧠 Smart KYC Risk Scoring Engine")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("### Raw Data", df)

    if "risk_label" in df.columns:
        st.success("Training model...")

        train_model(df)
        st.success("Model trained successfully!")
    else:
        st.warning("Upload data with 'risk_label' column to train model")

    if st.button("Run Risk Scoring"):
        result = predict(df)
        st.write("### Results", result)
