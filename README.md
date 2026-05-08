````markdown
# Loan Default Prediction – Monsoon Modelling Test

This repository contains my solution for the Monsoon Modelling Test, where the objective was to build a machine learning model to predict loan default risk using historical credit and enquiry data.

---

# Problem Statement

The client is a lending company that provided historical loan application data along with loan repayment outcomes. The goal is to predict whether a borrower will default on a loan at the time of application.

### Target Variable
- `TARGET = 1` → Bad Loan / Default
- `TARGET = 0` → Good Loan

### Evaluation Metric
- **ROC-AUC Score**

---

# Dataset Overview

The dataset consists of three major components:

## 1. Train/Test Flag Data
Contains:
- `uid`
- `NAME_CONTRACT_TYPE`
- `TARGET` (only in train)

---

## 2. Accounts Data (JSON)
Contains historical loan account information for each applicant.

### Important Fields
- `credit_type`
- `loan_amount`
- `amount_overdue`
- `open_date`
- `closed_date`
- `payment_hist_string`

### payment_hist_string
This field represents month-by-month repayment history.

Each 3 digits represent:
- number of days past due (DPD)

Example:

```text
000000030060
````

Meaning:

* On-time payments initially
* Then 30 days overdue
* Then 60 days overdue

The most recent repayment information is towards the right side of the string.

---

## 3. Enquiry Data (JSON)

Contains historical loan enquiries made by applicants.

### Important Fields

* `enquiry_type`
* `enquiry_amt`
* `enquiry_date`

---

# Approach

## Data Processing

* Flattened nested JSON structures
* Converted date fields into datetime format
* Aggregated multiple accounts and enquiries per customer (`uid`)
* Handled missing values and active loans

---

# Feature Engineering

The majority of the performance improvement came from feature engineering on repayment history.

## Repayment History Features

Created advanced delinquency-based features including:

* Maximum DPD
* Mean DPD
* Recent DPD
* 30/60/90+ DPD counts
* Number of late payments
* Consecutive late payment streaks
* Delinquency trend features
* Weighted recent delinquency
* Repayment volatility
* Recent vs historical repayment behavior

---

## Loan Features

* Total loan amount
* Mean/max loan amount
* Active loan amount
* Overdue ratios
* Loan duration
* Number of active loans
* Credit diversity

---

## Enquiry Features

* Number of enquiries
* Enquiry amount statistics
* Recent enquiry counts (30/90 days)
* Enquiry recency features
* Frequency encoded enquiry types

---

# Models Used

The following models were trained and evaluated using Stratified K-Fold Cross Validation:

* Logistic Regression
* LightGBM
* XGBoost
* CatBoost

Additionally, ensemble blending techniques were used to improve robustness.

---

# Validation Scores

| Model               | ROC-AUC |
| ------------------- | ------- |
| Logistic Regression | 0.6732  |
| LightGBM            | 0.6710  |
| XGBoost             | 0.6745  |
| Ensemble            | 0.6745  |

---

# Cross Validation Strategy

* Stratified 5-Fold Cross Validation
* Out-of-Fold predictions used for unbiased validation
* Evaluation Metric: ROC-AUC

---

# Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* LightGBM
* XGBoost
* CatBoost

---


# Key Learnings

* Sequential repayment history is highly predictive in credit-risk modelling.
* Recent delinquency patterns carry stronger signals than older repayment behavior.
* Feature engineering contributes significantly more than model complexity in tabular credit datasets.
* Ensemble methods help improve prediction stability.

---

# Final Submission

The final predictions are generated in the required competition format:

```text
final_submission_tejasv_gupta.csv
```

compatible with the provided `sample_submission.csv`.

---

