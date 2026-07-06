# Documentation — Loan Approval Predictor

## 1. Problem Statement
Financial institutions need to decide whether to approve or reject a loan application. This decision depends on multiple applicant factors such as income, credit history, education, and property location. Manually reviewing every application is slow and inconsistent. This project builds a Machine Learning model that predicts the loan approval outcome automatically, based on historical patterns in applicant data.

## 2. Methodology

### 2.1 Data Collection
A synthetic dataset of 800 loan applicant records was generated (`src/generate_dataset.py`), structured identically to the popular public "Loan Prediction Problem Dataset". Approval outcomes were generated using a rule-based scoring system (favoring good credit history and adequate income) combined with random noise, so the data reflects realistic, learnable patterns rather than being purely random.

### 2.2 Data Cleaning (`src/data_cleaning.py`)
Real-world data is rarely perfect. The raw dataset contains missing values in 7 columns. These were handled as follows:
- **Categorical columns** (Gender, Married, Dependents, Self_Employed): missing values filled with the **mode** (most frequent category).
- **Numerical columns** (LoanAmount, Loan_Amount_Term, Credit_History): missing values filled with the **median**, which is robust to outliers.
- A new feature, **TotalIncome** (ApplicantIncome + CoapplicantIncome), was engineered because combined household income is a stronger predictor than applicant income alone.

### 2.3 Exploratory Data Analysis & Visualization (`src/visualization.py`)
Six charts were generated to understand the data before modeling:
1. Loan Status Distribution — overall approval vs rejection balance
2. Total Income vs Loan Status — do higher earners get approved more?
3. Credit History Effect — the strongest predictor found (approval rate jumps from ~8% to ~95% with good credit history)
4. Education Effect on Approval
5. Property Area Effect on Approval
6. Correlation Heatmap of numerical features

### 2.4 Model Building (`src/train_model.py`)
- All categorical text columns were converted to numbers using **Label Encoding**.
- Data was split 80% training / 20% testing using stratified sampling (to preserve the approve/reject ratio in both sets).
- Two models were trained:
  - **Logistic Regression** — a simple linear baseline model, fast and interpretable.
  - **Random Forest Classifier** — an ensemble of decision trees, generally more accurate on tabular data.
- Both were evaluated using **Accuracy**, **Precision**, **Recall**, **F1-score**, and a **Confusion Matrix**.
- The Random Forest model performed best (~97.5% accuracy) and was saved as the final model.

### 2.5 Prediction Interface
Two ways are provided to use the trained model on new applicants:
- **`src/predict.py`** — a terminal-based tool that asks questions and prints the prediction.
- **`src/app.py`** — a Streamlit web dashboard with a form-based UI, live prediction, dataset overview, and all charts in one place.

## 3. Results Summary
| Model | Accuracy | Notes |
|---|---|---|
| Logistic Regression | ~96.9% | Fast, interpretable baseline |
| Random Forest Classifier | ~97.5% | Best model — selected as final |

The most influential feature identified was **Credit History**, followed by **Total Income** and **Loan Amount**.

## 4. Limitations & Future Improvements
- The dataset is synthetic; results on a real-world Kaggle dataset may vary slightly.
- Additional models (XGBoost, SVM) could be compared for further accuracy gains.
- Hyperparameter tuning (GridSearchCV) could further improve the Random Forest model.
- A real deployment would need to handle fairness/bias checks (e.g., ensuring gender does not unfairly influence predictions), which is an important consideration for real financial ML systems.

## 5. Tools & Technologies Used
Python 3, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, Joblib, Streamlit, Git & GitHub, Visual Studio Code.
