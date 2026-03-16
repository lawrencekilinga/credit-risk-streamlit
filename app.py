import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go

# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="Microcredit Risk Dashboard",
    page_icon="💳",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("credit_risk_model.pkl")

model = load_model()
feature_names = model.feature_names_in_

# --------------------------------------------------
# SECTOR AVERAGES (Replace with your real averages)
# --------------------------------------------------

sector_averages = {
    "SME": {
        "interest_rate_(apr)":0.32,
        "interest_burden":0.15,
        "outstanding_ratio":0.45,
        "arrears_ratio":0.08,
        "loan_age_days":210
    },
    "Micro": {
        "interest_rate_(apr)":0.42,
        "interest_burden":0.18,
        "outstanding_ratio":0.50,
        "arrears_ratio":0.12,
        "loan_age_days":180
    },
    "Consumer": {
        "interest_rate_(apr)":0.40,
        "interest_burden":0.17,
        "outstanding_ratio":0.48,
        "arrears_ratio":0.10,
        "loan_age_days":150
    },
    "Boda Boda": {
        "interest_rate_(apr)":0.36,
        "interest_burden":0.16,
        "outstanding_ratio":0.47,
        "arrears_ratio":0.09,
        "loan_age_days":200
    },
    "Mobile Money": {
        "interest_rate_(apr)":0.35,
        "interest_burden":0.14,
        "outstanding_ratio":0.44,
        "arrears_ratio":0.07,
        "loan_age_days":190
    },
    "Corporate": {
        "interest_rate_(apr)":0.25,
        "interest_burden":0.10,
        "outstanding_ratio":0.35,
        "arrears_ratio":0.03,
        "loan_age_days":250
    },
    "Express Motor": {
        "interest_rate_(apr)":0.30,
        "interest_burden":0.12,
        "outstanding_ratio":0.40,
        "arrears_ratio":0.05,
        "loan_age_days":220
    },
    "Micro Chap chap": {
        "interest_rate_(apr)":0.45,
        "interest_burden":0.20,
        "outstanding_ratio":0.52,
        "arrears_ratio":0.15,
        "loan_age_days":170
    },
    "TEST": {
        "interest_rate_(apr)":0.30,
        "interest_burden":0.12,
        "outstanding_ratio":0.40,
        "arrears_ratio":0.05,
        "loan_age_days":200
    }
}

# --------------------------------------------------
# DASHBOARD TITLE
# --------------------------------------------------

st.title("💳 Microcredit Credit Risk Dashboard")
st.markdown("Machine Learning powered **Loan Default Prediction Tool**")

# --------------------------------------------------
# SIDEBAR INPUTS
# --------------------------------------------------

st.sidebar.header("Loan Application")

loan_amount = st.sidebar.number_input("Loan Amount", min_value=0)

tenor = st.sidebar.number_input("Tenor (months)", min_value=1)

sector = st.sidebar.selectbox(
    "Sector",
    list(sector_averages.keys())
)

payment_frequency = st.sidebar.selectbox(
    "Payment Frequency",
    ["Weekly","Monthly"]
)

run_model = st.sidebar.button("Run Risk Assessment")

# --------------------------------------------------
# MODEL PREDICTION
# --------------------------------------------------

if run_model:

    input_data = pd.DataFrame(columns=feature_names)
    input_data.loc[0] = 0

    input_data["disbursed_amount"] = loan_amount
    input_data["tenor"] = tenor

    # Apply sector averages
    sector_vals = sector_averages.get(sector)

    input_data["interest_rate_(apr)"] = sector_vals["interest_rate_(apr)"]
    input_data["interest_burden"] = sector_vals["interest_burden"]
    input_data["outstanding_ratio"] = sector_vals["outstanding_ratio"]
    input_data["arrears_ratio"] = sector_vals["arrears_ratio"]
    input_data["loan_age_days"] = sector_vals["loan_age_days"]

    # Sector dummy variable
    sector_col = f"sector_{sector}"
    if sector_col in input_data.columns:
        input_data[sector_col] = 1

    # Payment frequency dummy
    if payment_frequency == "Weekly":
        if "payment_frequency_Weekly" in input_data.columns:
            input_data["payment_frequency_Weekly"] = 1

    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    risk_percent = int(probability * 100)

    credit_score = int((1 - probability) * 850)

    # --------------------------------------------------
    # METRIC CARDS
    # --------------------------------------------------

    col1, col2, col3 = st.columns(3)

    col1.metric("Loan Amount", f"{loan_amount:,.0f}")
    col2.metric("Credit Score", credit_score)
    col3.metric("Default Probability", f"{risk_percent}%")

    st.divider()

    # --------------------------------------------------
    # RISK GAUGE
    # --------------------------------------------------

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_percent,
        title={'text': "Default Risk (%)"},
        gauge={
            'axis': {'range': [0,100]},
            'bar': {'color': "darkred"},
            'steps': [
                {'range':[0,30],'color':"green"},
                {'range':[30,60],'color':"yellow"},
                {'range':[60,100],'color':"red"}
            ]
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------
    # EXPECTED LOSS
    # --------------------------------------------------

    sector_lgd = {
        "SME":0.45,
        "Micro":0.65,
        "Consumer":0.60,
        "Boda Boda":0.55,
        "Mobile Money":0.50,
        "Corporate":0.35,
        "Express Motor":0.50,
        "Micro Chap chap":0.70,
        "TEST":0.50
    }

    pd_value = probability
    lgd = sector_lgd.get(sector,0.5)
    ead = loan_amount

    expected_loss = pd_value * lgd * ead

    st.subheader("Credit Risk Metrics")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("PD", f"{pd_value:.2%}")
    c2.metric("LGD", f"{lgd:.0%}")
    c3.metric("Exposure", f"{ead:,.0f}")
    c4.metric("Expected Loss", f"{expected_loss:,.0f}")

    # --------------------------------------------------
    # ESTIMATED INTEREST
    # --------------------------------------------------

    estimated_interest = loan_amount * sector_vals["interest_rate_(apr)"] * (tenor/12)

    st.subheader("Estimated Loan Interest")

    st.metric("Estimated Interest", f"{estimated_interest:,.0f}")

    # --------------------------------------------------
    # LOAN DECISION
    # --------------------------------------------------

    st.subheader("Loan Recommendation")

    if risk_percent < 30:
        st.success("Loan Approved – Low Risk")
    elif risk_percent < 60:
        st.warning("Manual Review Recommended")
    else:
        st.error("Loan Rejected – High Risk")