"""
train_model.py
----------------
This script builds the PREDICTION MODEL for Loan Approval.

Steps performed:
1. Load the cleaned dataset
2. Encode categorical (text) columns into numbers using Label Encoding,
   because Machine Learning models only understand numbers.
3. Split the data into training set (80%) and testing set (20%)
4. Train TWO models so we can compare them:
      - Logistic Regression (simple, fast, interpretable baseline)
      - Random Forest Classifier (usually more accurate)
5. Evaluate both models using Accuracy, Confusion Matrix, and
   Classification Report (Precision / Recall / F1-score)
6. Save the BETTER performing model to model/loan_model.pkl using joblib,
   so it can be reused later without retraining (see predict.py)
7. Save evaluation results & confusion matrix images to outputs/

Run this AFTER data_cleaning.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

CLEAN_PATH = os.path.join("data", "loan_data_cleaned.csv")
MODEL_PATH = os.path.join("model", "loan_model.pkl")
ENCODERS_PATH = os.path.join("model", "label_encoders.pkl")
GRAPH_DIR = os.path.join("outputs", "graphs")
RESULTS_PATH = os.path.join("outputs", "model_results.txt")


def load_and_prepare_data():
    """Load cleaned data and encode categorical columns to numeric form."""
    df = pd.read_csv(CLEAN_PATH)

    # Loan_ID is just an identifier, not useful for prediction -> drop it
    df = df.drop(columns=["Loan_ID"])

    # Columns that contain text categories which need encoding
    categorical_cols = [
        "Gender", "Married", "Dependents", "Education",
        "Self_Employed", "Property_Area", "Loan_Status",
    ]

    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le  # Save encoder so predict.py can reverse the encoding later

    return df, encoders


def train_and_evaluate(df):
    """Split data, train both models, and return the results."""
    X = df.drop(columns=["Loan_Status"])   # Features (inputs)
    y = df["Loan_Status"]                  # Target (what we want to predict)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=5000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)

        results[name] = {
            "model": model,
            "accuracy": acc,
            "report": report,
            "confusion_matrix": cm,
        }
        print(f"\n{name} Accuracy: {acc:.4f}")
        print(report)

    return results, X.columns.tolist()


def save_confusion_matrix(cm, name):
    """Save a confusion matrix heatmap image."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Rejected", "Approved"],
                yticklabels=["Rejected", "Approved"], ax=ax)
    ax.set_title(f"Confusion Matrix - {name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    os.makedirs(GRAPH_DIR, exist_ok=True)
    safe_name = name.lower().replace(" ", "_")
    path = os.path.join(GRAPH_DIR, f"07_confusion_matrix_{safe_name}.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved chart: {path}")


def main():
    df, encoders = load_and_prepare_data()
    results, feature_names = train_and_evaluate(df)

    # Save confusion matrix images for both models
    for name, res in results.items():
        save_confusion_matrix(res["confusion_matrix"], name)

    # Pick the best model based on accuracy
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best_model = results[best_name]["model"]
    print(f"\nBest model: {best_name} (Accuracy: {results[best_name]['accuracy']:.4f})")

    # Save the best model + the label encoders + feature order to disk
    os.makedirs("model", exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump({"encoders": encoders, "feature_names": feature_names}, ENCODERS_PATH)
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Encoders saved to: {ENCODERS_PATH}")

    # Write a text summary of results (useful for README / documentation)
    os.makedirs("outputs", exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        f.write("LOAN APPROVAL PREDICTION - MODEL RESULTS\n")
        f.write("=" * 45 + "\n\n")
        for name, res in results.items():
            f.write(f"Model: {name}\n")
            f.write(f"Accuracy: {res['accuracy']:.4f}\n")
            f.write("Classification Report:\n")
            f.write(res["report"])
            f.write("\n" + "-" * 45 + "\n\n")
        f.write(f"BEST MODEL SELECTED: {best_name}\n")
    print(f"Results summary saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
