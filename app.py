import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load("credit_risk_model.pkl")

# Extract training feature names
feature_names = model.feature_names_in_

st.title("Microcredit Default Risk Prediction")

st.write("Enter borrower loan details to predict credit risk.")

# -------------------------
# USER INPUTS
# -------------------------

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

# -------------------------
# PREDICTION
# -------------------------

if st.button("Predict Default Risk"):

    # Create dataframe with all model features initialized to zero
    input_data = pd.DataFrame(columns=feature_names)
    input_data.loc[0] = 0

    # Fill user input features
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

    # Model prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # -------------------------
    # PREDICTION RESULT
    # -------------------------

    st.subheader("Default Risk Prediction")

    if prediction == 1:
        st.error(f"High Default Risk (Probability: {probability:.2f})")
    else:
        st.success(f"Low Default Risk (Probability: {probability:.2f})")

    # -------------------------
    # CREDIT RISK METER
    # -------------------------

    st.subheader("Credit Risk Meter")

    risk_percentage = int(probability * 100)

    st.progress(risk_percentage)

    st.write(f"Default Probability: {risk_percentage}%")

    if risk_percentage < 30:
        st.success("Low Risk Borrower")
    elif risk_percentage < 60:
        st.warning("Moderate Risk Borrower")
    else:
        st.error("High Risk Borrower")

    # -------------------------
    # INTEREST CALCULATION
    # -------------------------

    st.subheader("Estimated Interest Charged")

    sector_rates = {
        "Boda Boda": 0.36,
        "Consumer": 0.40,
        "Corporate": 0.25,
        "Express Motor": 0.30,
        "Micro": 0.42,
        "Micro Chap chap": 0.45,
        "Mobile Money": 0.35,
        "SME": 0.32,
        "TEST": 0.30
    }

    sector_rate = sector_rates.get(sector, 0.35)

    estimated_interest = disbursed_amount * sector_rate * (tenor / 12)

    st.write(f"Sector Interest Rate: {sector_rate*100:.1f}% APR")

    st.write(f"Estimated Interest Over Loan Period: **{estimated_interest:,.2f}**")

    # -------------------------
    # SIMPLE LOAN DECISION
    # -------------------------

    st.subheader("Loan Recommendation")

    if risk_percentage < 30:
        st.success("Loan Approved – Low Risk Borrower")
    elif risk_percentage < 60:
        st.warning("Loan Requires Manual Review")
    else:
        st.error("Loan Rejected – High Default Risk")