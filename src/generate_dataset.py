"""
generate_dataset.py
--------------------
This script CREATES a synthetic (dummy) Loan Approval dataset.

WHY a synthetic dataset?
Internship guidelines allow "dummy datasets or public datasets".
This script builds a realistic, Kaggle-style "Loan Prediction" dataset
using random number generation with sensible rules, so the project
does not depend on downloading a file from the internet.

If you prefer, you can later replace 'data/loan_data.csv' with a real
dataset downloaded from Kaggle (search "Loan Prediction Problem Dataset")
as long as the column names match, or you adjust the other scripts.

Run this file first. It creates: data/loan_data.csv
"""

import numpy as np
import pandas as pd
import os

# Setting a random seed makes the "random" data reproducible.
# Anyone who runs this script will get the exact same dataset.
np.random.seed(42)

# Number of loan applicant records to generate
NUM_RECORDS = 800

# ---------------------------------------------------------
# STEP 1: Generate each column using realistic distributions
# ---------------------------------------------------------

loan_id = [f"LP{1000 + i}" for i in range(NUM_RECORDS)]

gender = np.random.choice(["Male", "Female"], size=NUM_RECORDS, p=[0.65, 0.35])

married = np.random.choice(["Yes", "No"], size=NUM_RECORDS, p=[0.65, 0.35])

dependents = np.random.choice(["0", "1", "2", "3+"], size=NUM_RECORDS,
                               p=[0.55, 0.20, 0.15, 0.10])

education = np.random.choice(["Graduate", "Not Graduate"], size=NUM_RECORDS,
                              p=[0.78, 0.22])

self_employed = np.random.choice(["Yes", "No"], size=NUM_RECORDS, p=[0.15, 0.85])

# Income figures (in currency units) using a log-normal distribution
# so most incomes are moderate but a few are very high (realistic skew).
applicant_income = np.round(np.random.lognormal(mean=8.4, sigma=0.5, size=NUM_RECORDS))
coapplicant_income = np.round(np.random.lognormal(mean=7.5, sigma=0.9, size=NUM_RECORDS))
# Some applicants have no co-applicant income at all
mask_no_coapplicant = np.random.rand(NUM_RECORDS) < 0.35
coapplicant_income[mask_no_coapplicant] = 0

# Loan amount (in thousands), roughly related to combined income
combined_income = applicant_income + coapplicant_income
loan_amount = np.round((combined_income / 90) + np.random.normal(0, 25, NUM_RECORDS))
loan_amount = np.clip(loan_amount, 10, 700)

loan_amount_term = np.random.choice(
    [360, 180, 240, 120, 60, 300, 84], size=NUM_RECORDS,
    p=[0.65, 0.10, 0.08, 0.06, 0.05, 0.03, 0.03]
)

# Credit history: 1 = good credit history, 0 = bad/no credit history
credit_history = np.random.choice([1, 0], size=NUM_RECORDS, p=[0.82, 0.18])

property_area = np.random.choice(["Urban", "Semiurban", "Rural"], size=NUM_RECORDS,
                                  p=[0.38, 0.38, 0.24])

# ---------------------------------------------------------
# STEP 2: Decide Loan_Status using a rule + randomness
# ---------------------------------------------------------
# This mimics real-world approval logic:
#   - Good credit history strongly increases approval chance
#   - Higher combined income increases approval chance
#   - Very high loan amount relative to income decreases approval chance
#   - A little random noise is added so the model has something to "learn"

approval_score = (
    (credit_history == 1) * 3.0
    + (combined_income > 6000) * 1.0
    + (education == "Graduate") * 0.5
    - (loan_amount > (combined_income / 40)) * 1.2
    + np.random.normal(0, 1.0, NUM_RECORDS)  # random noise
)

loan_status = np.where(approval_score > 2.0, "Y", "N")

# ---------------------------------------------------------
# STEP 3: Assemble into a DataFrame
# ---------------------------------------------------------
df = pd.DataFrame({
    "Loan_ID": loan_id,
    "Gender": gender,
    "Married": married,
    "Dependents": dependents,
    "Education": education,
    "Self_Employed": self_employed,
    "ApplicantIncome": applicant_income.astype(int),
    "CoapplicantIncome": coapplicant_income.astype(int),
    "LoanAmount": loan_amount.astype(int),
    "Loan_Amount_Term": loan_amount_term,
    "Credit_History": credit_history,
    "Property_Area": property_area,
    "Loan_Status": loan_status,
})

# ---------------------------------------------------------
# STEP 4: Deliberately introduce some missing values
# ---------------------------------------------------------
# Real-world datasets almost always have missing values.
# We inject a few so the "Data Cleaning" step has real work to do.
for col in ["Gender", "Married", "Dependents", "Self_Employed",
            "LoanAmount", "Loan_Amount_Term", "Credit_History"]:
    missing_idx = np.random.choice(df.index, size=int(0.03 * NUM_RECORDS), replace=False)
    df.loc[missing_idx, col] = np.nan

# ---------------------------------------------------------
# STEP 5: Save to CSV inside the data/ folder
# ---------------------------------------------------------
os.makedirs("data", exist_ok=True)
output_path = os.path.join("data", "loan_data.csv")
df.to_csv(output_path, index=False)

print(f"Dataset created successfully: {output_path}")
print(f"Shape of dataset: {df.shape}")
print(df.head())
