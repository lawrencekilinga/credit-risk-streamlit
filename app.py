import streamlit as st
import joblib
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Microcredit Risk Dashboard",
    page_icon="💳",
    layout="wide"
)

# Load model
model = joblib.load("credit_risk_model.pkl")
feature_names = model.feature_names_in_

# Title
st.title("💳 Microcredit Risk Assessment Dashboard")
st.markdown("Machine Learning powered **Loan Default Prediction System**")

# -------------------------
# SIDEBAR INPUTS
# -------------------------

st.sidebar.header("Loan Details")

disbursed_amount = st.sidebar.number_input("Loan Amount", min_value=0)

tenor = st.sidebar.number_input("Loan Tenor (months)", min_value=1)

sector = st.sidebar.selectbox(
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

payment_frequency = st.sidebar.selectbox(
    "Payment Frequency",
    ["Weekly", "Monthly"]
)

predict_button = st.sidebar.button("Run Credit Risk Assessment")

# -------------------------
# PREDICTION
# -------------------------

if predict_button:

    input_data = pd.DataFrame(columns=feature_names)
    input_data.loc[0] = 0

    if "disbursed_amount" in input_data.columns:
        input_data["disbursed_amount"] = disbursed_amount

    if "tenor" in input_data.columns:
        input_data["tenor"] = tenor

    sector_column = f"sector_{sector}"
    if sector_column in input_data.columns:
        input_data[sector_column] = 1

    if payment_frequency == "Weekly":
        if "payment_frequency_Weekly" in input_data.columns:
            input_data["payment_frequency_Weekly"] = 1

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    risk_percent = int(probability * 100)

    # -------------------------
    # TOP METRICS
    # -------------------------

    col1, col2, col3 = st.columns(3)

    col1.metric("Loan Amount", f"{disbursed_amount:,.0f}")
    col2.metric("Tenor (months)", tenor)
    col3.metric("Default Probability", f"{risk_percent}%")

    st.divider()

    # -------------------------
    # CREDIT SCORE
    # -------------------------

    credit_score = int((1 - probability) * 850)

    st.subheader("Credit Score")

    st.metric(
        label="Borrower Credit Score",
        value=credit_score
    )

    # -------------------------
    # RISK METER
    # -------------------------

    st.subheader("Credit Risk Meter")

    st.progress(risk_percent)

    if risk_percent < 30:
        st.success("Low Risk Borrower")
    elif risk_percent < 60:
        st.warning("Moderate Risk Borrower")
    else:
        st.error("High Default Risk")

    # -------------------------
    # INTEREST CALCULATION
    # -------------------------

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

    st.subheader("Estimated Loan Interest")

    col1, col2 = st.columns(2)

    col1.metric("Sector APR", f"{sector_rate*100:.1f}%")
    col2.metric("Estimated Interest", f"{estimated_interest:,.0f}")

    # -------------------------
    # LOAN DECISION
    # -------------------------

    st.subheader("Loan Recommendation")

    if risk_percent < 30:
        st.success("Loan Approved")
    elif risk_percent < 60:
        st.warning("Manual Credit Review Required")
    else:
        st.error("Loan Rejected – High Risk")