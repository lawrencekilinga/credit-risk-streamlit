import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load("credit_risk_model.pkl")

# Extract model feature names
feature_names = model.feature_names_in_

st.title("Microcredit Default Risk Prediction")

st.write("Enter borrower loan details to predict default risk.")

# User Inputs
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
    ["Weekly", "Monthly"]
)

# Prediction button
if st.button("Predict Default Risk"):

    # Create dataframe with all model features initialized to 0
    input_data = pd.DataFrame(columns=feature_names)
    input_data.loc[0] = 0

    # Fill the features we collected
    if "disbursed_amount" in input_data.columns:
        input_data["disbursed_amount"] = disbursed_amount

    if "tenor" in input_data.columns:
        input_data["tenor"] = tenor

    # Sector encoding
    sector_column = f"sector_{sector}"
    if sector_column in input_data.columns:
        input_data[sector_column] = 1

    # Payment frequency encoding
    if payment_frequency == "Weekly":
        if "payment_frequency_Weekly" in input_data.columns:
            input_data["payment_frequency_Weekly"] = 1

    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Display results
    if prediction == 1:
        st.error(f"High Default Risk (Probability: {probability:.2f})")
    else:
        st.success(f"Low Default Risk (Probability: {probability:.2f})")

    st.write("Prediction Probability:", round(probability, 3))