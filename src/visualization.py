"""
visualization.py
------------------
This script performs DATA VISUALIZATION on the cleaned loan dataset.

It creates several charts to help understand the data before building
a prediction model, and saves each chart as a .png image inside
outputs/graphs/ so they can be used as "Output Screenshots" for the
project submission.

Run this AFTER data_cleaning.py
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Allows saving images without opening a display window
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("whitegrid")

CLEAN_PATH = os.path.join("data", "loan_data_cleaned.csv")
GRAPH_DIR = os.path.join("outputs", "graphs")


def save_fig(fig, name):
    """Helper function to save a matplotlib figure to the graphs folder."""
    os.makedirs(GRAPH_DIR, exist_ok=True)
    path = os.path.join(GRAPH_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved chart: {path}")


def plot_loan_status_distribution(df):
    """Bar chart: how many loans were Approved (Y) vs Rejected (N)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x="Loan_Status", data=df, palette="viridis", ax=ax)
    ax.set_title("Loan Approval Status Distribution")
    ax.set_xlabel("Loan Status (Y = Approved, N = Rejected)")
    ax.set_ylabel("Number of Applicants")
    save_fig(fig, "01_loan_status_distribution.png")


def plot_income_vs_status(df):
    """Boxplot: total income comparison between approved and rejected loans."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x="Loan_Status", y="TotalIncome", data=df, palette="Set2", ax=ax)
    ax.set_title("Total Income vs Loan Status")
    ax.set_ylim(0, df["TotalIncome"].quantile(0.95))  # remove extreme outliers visually
    save_fig(fig, "02_income_vs_loan_status.png")


def plot_credit_history_effect(df):
    """Bar chart: approval rate grouped by credit history."""
    fig, ax = plt.subplots(figsize=(6, 4))
    approval_rate = df.groupby("Credit_History")["Loan_Status"].apply(
        lambda x: (x == "Y").mean() * 100
    )
    approval_rate.plot(kind="bar", color=["#e74c3c", "#2ecc71"], ax=ax)
    ax.set_title("Loan Approval Rate by Credit History")
    ax.set_xlabel("Credit History (0 = Bad, 1 = Good)")
    ax.set_ylabel("Approval Rate (%)")
    ax.set_xticklabels(["0", "1"], rotation=0)
    save_fig(fig, "03_credit_history_effect.png")


def plot_education_effect(df):
    """Stacked bar chart: loan status by education level."""
    fig, ax = plt.subplots(figsize=(6, 4))
    pd.crosstab(df["Education"], df["Loan_Status"]).plot(
        kind="bar", stacked=True, color=["#e74c3c", "#2ecc71"], ax=ax
    )
    ax.set_title("Loan Status by Education Level")
    ax.set_xlabel("Education")
    ax.set_ylabel("Number of Applicants")
    ax.legend(title="Loan Status")
    save_fig(fig, "04_education_effect.png")


def plot_property_area_effect(df):
    """Bar chart: approval rate by property area."""
    fig, ax = plt.subplots(figsize=(6, 4))
    approval_rate = df.groupby("Property_Area")["Loan_Status"].apply(
        lambda x: (x == "Y").mean() * 100
    )
    approval_rate.plot(kind="bar", color="#3498db", ax=ax)
    ax.set_title("Loan Approval Rate by Property Area")
    ax.set_ylabel("Approval Rate (%)")
    ax.set_xticklabels(approval_rate.index, rotation=0)
    save_fig(fig, "05_property_area_effect.png")


def plot_correlation_heatmap(df):
    """Heatmap: correlation between numerical features."""
    fig, ax = plt.subplots(figsize=(7, 5))
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap of Numerical Features")
    save_fig(fig, "06_correlation_heatmap.png")


def main():
    df = pd.read_csv(CLEAN_PATH)
    plot_loan_status_distribution(df)
    plot_income_vs_status(df)
    plot_credit_history_effect(df)
    plot_education_effect(df)
    plot_property_area_effect(df)
    plot_correlation_heatmap(df)
    print("\nAll charts generated successfully in outputs/graphs/")


if __name__ == "__main__":
    main()
