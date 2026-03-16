import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("credit_risk_model.pkl")

st.title("Microcredit Default Risk Prediction")

st.write("Enter borrower loan details.")

# Inputs
disbursed_amount = st.number_input("Loan Amount", min_value=0)

tenor = st.number_input("Loan Tenor (months)", min_value=1)

sector = st.selectbox(
    "Sector",
    [
        "Boda Boda",
        "Consumer",
        "Corporate",
        "Express Motor",
        "Micro",
        "Micro Chap chap",
        "Mobile Money",
        "SME",
        "TEST"
    ]
)

payment_frequency = st.selectbox(
    "Payment Frequency",
    ["Weekly","Monthly"]
)

if st.button("Predict Default Risk"):

    # Sector dummy variables
    sector_features = {
        "sector_Boda Boda":0,
        "sector_Consumer":0,
        "sector_Corporate":0,
        "sector_Express Motor":0,
        "sector_Micro":0,
        "sector_Micro Chap chap":0,
        "sector_Mobile Money":0,
        "sector_SME":0,
        "sector_TEST":0
    }

    sector_features[f"sector_{sector}"] = 1

    # Payment frequency dummy
    payment_freq_weekly = 1 if payment_frequency == "Weekly" else 0

    # Construct input dataframe
    input_data = pd.DataFrame({
        "disbursed_amount":[disbursed_amount],
        "outstanding_balance_(p+i)":[0],
        "principal":[0],
        "interest":[0],
        "interest_rate_(apr)":[0],
        "tenor":[tenor],
        "loan_age_days":[0],
        "outstanding_ratio":[0],
        "interest_burden":[0],
        "payment_frequency_Weekly":[payment_freq_weekly],
        **sector_features
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"High Default Risk (Probability: {probability:.2f})")
    else:
        st.success(f"Low Default Risk (Probability: {probability:.2f})")