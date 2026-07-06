"""
predict.py
-----------
This script lets you use the TRAINED MODEL to predict loan approval
for a NEW applicant, by typing answers into the terminal.

It loads:
    model/loan_model.pkl        -> the trained Random Forest / Logistic Regression model
    model/label_encoders.pkl    -> the encoders used to convert text to numbers

Run this AFTER train_model.py, using:
    python src/predict.py
"""

import joblib
import pandas as pd
import os

MODEL_PATH = os.path.join("model", "loan_model.pkl")
ENCODERS_PATH = os.path.join("model", "label_encoders.pkl")


def get_user_input():
    """Ask the user questions in the terminal to build one applicant record."""
    print("Please enter the applicant's details:\n")

    gender = input("Gender (Male/Female): ").strip()
    married = input("Married (Yes/No): ").strip()
    dependents = input("Number of Dependents (0/1/2/3+): ").strip()
    education = input("Education (Graduate/Not Graduate): ").strip()
    self_employed = input("Self Employed (Yes/No): ").strip()
    applicant_income = float(input("Applicant Monthly Income: ").strip())
    coapplicant_income = float(input("Co-applicant Monthly Income (0 if none): ").strip())
    loan_amount = float(input("Loan Amount Requested (in thousands): ").strip())
    loan_amount_term = float(input("Loan Term in days (e.g. 360): ").strip())
    credit_history = int(input("Credit History (1 = Good, 0 = Bad): ").strip())
    property_area = input("Property Area (Urban/Semiurban/Rural): ").strip()

    total_income = applicant_income + coapplicant_income

    data = {
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
    return data


def encode_input(data, encoders, feature_names):
    """Convert the raw text input into the numeric format the model expects."""
    df = pd.DataFrame([data])

    categorical_cols = ["Gender", "Married", "Dependents", "Education",
                         "Self_Employed", "Property_Area"]
    for col in categorical_cols:
        le = encoders[col]
        # Handle any unseen category safely by defaulting to the most common class
        df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
        df[col] = le.transform(df[col])

    # Make sure column order matches exactly what the model was trained on
    df = df[feature_names]
    return df


def main():
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Please run 'python src/train_model.py' first.")
        return

    model = joblib.load(MODEL_PATH)
    saved = joblib.load(ENCODERS_PATH)
    encoders = saved["encoders"]
    feature_names = saved["feature_names"]

    data = get_user_input()
    input_df = encode_input(data, encoders, feature_names)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    status_encoder = encoders["Loan_Status"]
    predicted_label = status_encoder.inverse_transform([prediction])[0]

    print("\n" + "=" * 40)
    if predicted_label == "Y":
        print("RESULT: Loan Approved ✅")
    else:
        print("RESULT: Loan Rejected ❌")
    print(f"Confidence: {max(probability) * 100:.2f}%")
    print("=" * 40)


if __name__ == "__main__":
    main()
