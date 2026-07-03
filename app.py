import streamlit as st
import pandas as pd
import joblib

# ======================================
# Page Configuration
# ======================================

st.set_page_config(
    page_title="Loan Approval Prediction System",
    page_icon="🏦",
    layout="wide"
)

# ======================================
# Load Model and Scaler
# ======================================

model = joblib.load("models/loan_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# ======================================
# Sidebar Information
# ======================================

st.sidebar.title("Project Information")

st.sidebar.info("""
### Machine Learning Model
- Logistic Regression

### Dataset
- Loan Approval Dataset

### Features Used
- Gender
- Marital Status
- Dependents
- Education
- Self Employed
- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Term
- Credit History
- Property Area
""")

# ======================================
# Title
# ======================================

st.title("🏦 Loan Approval Prediction System")

st.write(
    "Enter applicant details below to predict "
    "whether the loan application is likely "
    "to be approved or rejected."
)

# ======================================
# Input Section
# ======================================

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    married = st.selectbox(
        "Marital Status",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        [0, 1, 2, 3]
    )

    education = st.selectbox(
        "Education",
        ["Graduate", "Not Graduate"]
    )

    self_employed = st.selectbox(
        "Self Employed",
        ["Yes", "No"]
    )

    property_area = st.selectbox(
        "Property Area",
        ["Rural", "Semiurban", "Urban"]
    )

with col2:

    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0,
        value=5000,
        step=500
    )

    coapplicant_income = st.number_input(
        "Coapplicant Income",
        min_value=0,
        value=0,
        step=500
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0,
        value=120,
        step=10
    )

    loan_term = st.selectbox(
        "Loan Amount Term (Months)",
        [120, 180, 240, 300, 360]
    )

    credit_history = st.selectbox(
        "Credit History",
        [1, 0],
        format_func=lambda x:
        "Good Credit History" if x == 1
        else "Poor Credit History"
    )

# ======================================
# Prediction Button
# ======================================

if st.button("Predict Loan Status"):

    input_data = pd.DataFrame([[
        1 if gender == "Male" else 0,
        1 if married == "Yes" else 0,
        dependents,
        1 if education == "Graduate" else 0,
        1 if self_employed == "Yes" else 0,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        credit_history,
        {
            "Rural": 0,
            "Semiurban": 1,
            "Urban": 2
        }[property_area]
    ]], columns=[
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
        "Property_Area"
    ])

    # Scale input data
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    # Probability of approval
    probability = model.predict_proba(input_scaled)[0][1]

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(
            f"✅ Loan Approved"
        )
        st.info(
            f"Approval Probability: {probability * 100:.2f}%"
        )

    else:
        st.error(
            f"❌ Loan Rejected"
        )
        st.warning(
            f"Approval Probability: {probability * 100:.2f}%"
        )

    st.subheader("Applicant Summary")
    st.dataframe(input_data)