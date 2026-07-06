"""
app.py
-------
This is an interactive DASHBOARD built with Streamlit.

It satisfies the "Dashboard Creation" requirement of the internship
guidelines. It lets anyone:
  1. View the dataset and key statistics
  2. See the visualization charts
  3. Enter a new applicant's details in a form
  4. Get an instant Loan Approved / Rejected prediction with confidence %

HOW TO RUN THIS DASHBOARD:
    1. Make sure you have already run (once):
         python src/generate_dataset.py
         python src/data_cleaning.py
         python src/train_model.py
    2. Then run:
         streamlit run src/app.py
    3. It will open automatically in your browser at http://localhost:8501
"""

import streamlit as st
import pandas as pd
import joblib
import os

# ----------------------------------------------------------------
# Page configuration (must be the first Streamlit command used)
# ----------------------------------------------------------------
st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="wide")

DATA_PATH = os.path.join("data", "loan_data_cleaned.csv")
MODEL_PATH = os.path.join("model", "loan_model.pkl")
ENCODERS_PATH = os.path.join("model", "label_encoders.pkl")
GRAPH_DIR = os.path.join("outputs", "graphs")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    saved = joblib.load(ENCODERS_PATH)
    return model, saved["encoders"], saved["feature_names"]


st.title("🏦 Loan Approval Predictor")
st.caption("A Machine Learning powered dashboard to predict loan approval outcomes.")

tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "📈 Visualizations", "🔮 Predict Loan Approval"])

# ----------------------------------------------------------------
# TAB 1: DATA OVERVIEW
# ----------------------------------------------------------------
with tab1:
    st.subheader("Cleaned Dataset")
    if os.path.exists(DATA_PATH):
        df = load_data()
        st.dataframe(df.head(20), use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Applicants", len(df))
        col2.metric("Approved Loans", int((df["Loan_Status"] == "Y").sum()))
        col3.metric("Rejected Loans", int((df["Loan_Status"] == "N").sum()))

        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
    else:
        st.warning("Dataset not found. Please run the src/ scripts first (see README).")

# ----------------------------------------------------------------
# TAB 2: VISUALIZATIONS
# ----------------------------------------------------------------
with tab2:
    st.subheader("Generated Charts")
    if os.path.exists(GRAPH_DIR):
        images = sorted(
            f for f in os.listdir(GRAPH_DIR) if f.lower().endswith(".png")
        )
        if images:
            cols = st.columns(2)
            for i, img_name in enumerate(images):
                with cols[i % 2]:
                    st.image(os.path.join(GRAPH_DIR, img_name), caption=img_name, use_container_width=True)
        else:
            st.warning("No charts found. Run src/visualization.py first.")
    else:
        st.warning("Charts folder not found. Run src/visualization.py first.")

# ----------------------------------------------------------------
# TAB 3: LIVE PREDICTION FORM
# ----------------------------------------------------------------
with tab3:
    st.subheader("Enter Applicant Details")

    if not os.path.exists(MODEL_PATH):
        st.error("Trained model not found. Please run src/train_model.py first.")
    else:
        model, encoders, feature_names = load_model()

        with st.form("prediction_form"):
            c1, c2 = st.columns(2)
            with c1:
                gender = st.selectbox("Gender", ["Male", "Female"])
                married = st.selectbox("Married", ["Yes", "No"])
                dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
                education = st.selectbox("Education", ["Graduate", "Not Graduate"])
                self_employed = st.selectbox("Self Employed", ["Yes", "No"])
                property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
            with c2:
                applicant_income = st.number_input("Applicant Monthly Income", min_value=0, value=5000)
                coapplicant_income = st.number_input("Co-applicant Monthly Income", min_value=0, value=0)
                loan_amount = st.number_input("Loan Amount Requested (in thousands)", min_value=0, value=120)
                loan_amount_term = st.selectbox("Loan Term (days)", [360, 180, 240, 120, 60, 300, 84])
                credit_history = st.selectbox("Credit History", [1, 0], format_func=lambda x: "Good (1)" if x == 1 else "Bad (0)")

            submitted = st.form_submit_button("Predict Loan Status")

        if submitted:
            total_income = applicant_income + coapplicant_income
            input_data = {
                "Gender": gender,
                "Married": married,
                "Dependents": dependents,
                "Education": education,
                "Self_Employed": self_employed,
                "ApplicantIncome": applicant_income,
                "CoapplicantIncome": coapplicant_income,
                "LoanAmount": loan_amount,
                "Loan_Amount_Term": loan_amount_term,
                "Credit_History": credit_history,
                "Property_Area": property_area,
                "TotalIncome": total_income,
            }

            input_df = pd.DataFrame([input_data])
            categorical_cols = ["Gender", "Married", "Dependents", "Education",
                                 "Self_Employed", "Property_Area"]
            for col in categorical_cols:
                le = encoders[col]
                input_df[col] = input_df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                input_df[col] = le.transform(input_df[col])

            input_df = input_df[feature_names]

            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0]
            predicted_label = encoders["Loan_Status"].inverse_transform([prediction])[0]

            if predicted_label == "Y":
                st.success(f"✅ Loan Approved — Confidence: {max(probability) * 100:.2f}%")
            else:
                st.error(f"❌ Loan Rejected — Confidence: {max(probability) * 100:.2f}%")

st.markdown("---")
st.caption("Built with Python, scikit-learn & Streamlit | Loan Approval Predictor Project")
