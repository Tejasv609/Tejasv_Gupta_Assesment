# Loan Default Prediction – Monsoon Modelling Test

## Overview

This repository contains my solution for the **Monsoon Modelling Test**, where the objective was to build a machine learning model capable of predicting loan default risk using historical credit and enquiry data.

The project focuses heavily on:
- Credit risk modelling
- Feature engineering
- Sequential repayment analysis
- Ensemble learning

---

## Problem Statement

A lending company provided historical loan application data along with repayment outcomes. The task was to predict whether a borrower would default on a loan at the time of application.

### Target Variable

| Value | Meaning |
|---|---|
| 1 | Bad Loan / Default |
| 0 | Good Loan |

### Evaluation Metric

- **ROC-AUC Score**

---

## Dataset Description

The dataset consists of three major components:

### 1. Train/Test Flag Data

Contains:
- `uid`
- `NAME_CONTRACT_TYPE`
- `TARGET` (train only)

---

### 2. Accounts Data (JSON)

Contains historical loan account information for each applicant.

#### Important Fields

| Field | Description |
|---|---|
| `credit_type` | Type of previous loan |
| `loan_amount` | Principal loan amount |
| `amount_overdue` | Overdue amount |
| `open_date` | Loan start date |
| `closed_date` | Loan closure date |
| `payment_hist_string` | Monthly repayment history |

---

### payment_hist_string

This field represents month-by-month repayment behavior.

Each 3 digits represent:
- Number of days past due (DPD)

Example:

```text
000000030060
```

Meaning:
- On-time payments initially
- Then 30 days overdue
- Then 60 days overdue

The most recent repayment information appears towards the right side of the string.

---

### 3. Enquiry Data (JSON)

Contains historical loan enquiries made by applicants.

#### Important Fields

| Field | Description |
|---|---|
| `enquiry_type` | Type of loan enquiry |
| `enquiry_amt` | Requested amount |
| `enquiry_date` | Date of enquiry |

---

## Data Processing

The following preprocessing steps were performed:

- Flattened nested JSON structures
- Converted date columns into datetime format
- Aggregated multiple accounts and enquiries per customer
- Handled missing values
- Identified active loans
- Created temporal and repayment-based features

---

## Feature Engineering

The majority of the model performance improvement came from advanced feature engineering.

### Repayment History Features

Created features such as:

- Maximum DPD
- Mean DPD
- Recent DPD
- 30/60/90+ DPD counts
- Number of late payments
- Consecutive late payment streaks
- Repayment trend features
- Weighted recent delinquency
- Repayment volatility
- Recent vs historical delinquency behavior

---

### Loan Features

- Total loan amount
- Mean and maximum loan amount
- Active loan amount
- Overdue ratios
- Loan duration
- Number of active loans
- Credit diversity

---

### Enquiry Features

- Number of enquiries
- Enquiry amount statistics
- Recent enquiry counts (30/90 days)
- Enquiry recency features
- Frequency encoded enquiry types

---

## Models Used

The following models were trained and evaluated using Stratified K-Fold Cross Validation:

- Logistic Regression
- LightGBM
- XGBoost
- CatBoost

Ensemble blending techniques were also used to improve model robustness.

---

## Validation Scores

| Model | ROC-AUC |
|---|---|
| Logistic Regression | 0.6732 |
| LightGBM | 0.6710 |
| XGBoost | 0.6745 |
| Ensemble | 0.6745 |

---

## Cross Validation Strategy

- Stratified 5-Fold Cross Validation
- Out-of-Fold predictions for unbiased validation
- ROC-AUC used as the evaluation metric

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- LightGBM
- XGBoost
- CatBoost

---

## Repository Structure

```text
├── notebooks/
│   └── loan_default_v2.ipynb
│
├── final_submission/
│   └── final_submission_tejasv_gupta.csv
│
├── README.md
```

---

## Key Learnings

- Sequential repayment history is highly predictive in credit risk modelling.
- Recent delinquency patterns are stronger indicators than historical averages.
- Feature engineering contributes more than model complexity in tabular datasets.
- Ensemble methods improve prediction stability.

---

## Final Submission

The final predictions are generated in the required competition format:

```text
final_submission_tejasv_gupta.csv
```

compatible with the provided `sample_submission.csv`.

---

## Author

**Tejasv Gupta**
