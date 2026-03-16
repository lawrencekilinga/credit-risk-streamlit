import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("credit_risk_model.pkl")

st.title("Microcredit Default Risk Prediction")

st.write("Enter loan details to predict default risk.")

# Input fields
disbursed_amount = st.number_input("Loan Amount", min_value=0)
interest_rate = st.number_input("Interest Rate (%)", min_value=0.0)
tenor = st.number_input("Loan Tenor (months)", min_value=1)

outstanding_ratio = st.number_input("Outstanding Ratio", min_value=0.0)
interest_burden = st.number_input("Interest Burden", min_value=0.0)

# Prediction button
if st.button("Predict Default Risk"):

    input_data = pd.DataFrame({
        "disbursed_amount":[disbursed_amount],
        "interest_rate_(apr)":[interest_rate],
        "tenor":[tenor],
        "outstanding_ratio":[outstanding_ratio],
        "interest_burden":[interest_burden]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"High Default Risk (Probability: {probability:.2f})")
    else:
        st.success(f"Low Default Risk (Probability: {probability:.2f})")