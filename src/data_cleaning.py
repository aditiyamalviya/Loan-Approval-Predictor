"""
data_cleaning.py
-----------------
This script performs DATA CLEANING on the raw loan dataset.

What it does:
1. Loads the raw CSV from data/loan_data.csv
2. Reports how many missing values exist in each column
3. Fills (imputes) missing values using sensible strategies:
      - Categorical columns  -> filled with the most frequent value (mode)
      - Numerical columns    -> filled with the median (robust to outliers)
4. Fixes data types where needed
5. Saves a cleaned version to data/loan_data_cleaned.csv

Run this AFTER generate_dataset.py and BEFORE visualization.py / train_model.py
"""

import pandas as pd
import os

RAW_PATH = os.path.join("data", "loan_data.csv")
CLEAN_PATH = os.path.join("data", "loan_data_cleaned.csv")


def load_data(path):
    """Load the CSV file into a pandas DataFrame."""
    df = pd.read_csv(path)
    print(f"Loaded data from {path} -> shape: {df.shape}")
    return df


def report_missing_values(df):
    """Print a summary of missing values per column."""
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print("\nMissing values BEFORE cleaning:")
    if missing.empty:
        print("No missing values found.")
    else:
        print(missing)
    return missing


def clean_data(df):
    """Fill missing values and fix data types."""

    # --- Categorical columns: fill with the mode (most common value) ---
    categorical_cols = ["Gender", "Married", "Dependents", "Self_Employed"]
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_value = df[col].mode()[0]
            df[col] = df[col].fillna(mode_value)

    # --- Numerical columns: fill with the median ---
    numerical_cols = ["LoanAmount", "Loan_Amount_Term", "Credit_History"]
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)

    # --- Fix data types ---
    df["Credit_History"] = df["Credit_History"].astype(int)
    df["Loan_Amount_Term"] = df["Loan_Amount_Term"].astype(int)
    df["LoanAmount"] = df["LoanAmount"].astype(int)

    # --- Feature engineering: create a useful new column ---
    # Total household income = applicant + co-applicant income.
    # This is a common and meaningful feature for loan approval prediction.
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]

    return df


def main():
    df = load_data(RAW_PATH)
    report_missing_values(df)

    df_clean = clean_data(df)

    print("\nMissing values AFTER cleaning:")
    print(df_clean.isnull().sum().sum(), "total missing values remain.")

    os.makedirs("data", exist_ok=True)
    df_clean.to_csv(CLEAN_PATH, index=False)
    print(f"\nCleaned dataset saved to: {CLEAN_PATH}")
    print(f"Final shape: {df_clean.shape}")


if __name__ == "__main__":
    main()
