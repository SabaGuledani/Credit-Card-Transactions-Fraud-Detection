# Credit Card Fraud Analytics - Project Report

## 1) Project Overview

In this project, I analyzed credit card transaction data to understand fraud patterns from a business point of view.  
I built this work mainly as a **data analytics report**, and I added a small machine learning part that I am still improving.

My main objective was to answer practical questions:
- How often does fraud happen?
- How much money is lost because of fraud?
- Which customer and transaction segments show higher risk?

## 2) Dataset and Scope

- Raw dataset used: `data/raw/fraudTrain.csv`
- Processed dataset I created: `data/processed/cleeaned_fraud_train.csv`
- Target column: `is_fraud`

I used `notebooks/data_load.ipynb` for data preparation and feature engineering, then `notebooks/core_business_metrics.ipynb` for KPI analysis.

## 3) What I Did in Data Preparation

In `notebooks/data_load.ipynb`, I cleaned the data and engineered metrics that can support reporting and fraud monitoring.

### Time-based metrics
- `week_day`
- `hour`
- `trans_date`

### Customer profile metrics
- `age`
- `age_group`
- `city_type`

### Distance and location behavior
- `distance_from_home`
- `distance_group`
- `distance_from_prev` (distance between current and previous merchant location)

### Transaction behavior and anomaly metrics
- `time_since_prev_transaction`
- `time_since_prev_transaction_hours`
- `speed`, `speed_group`
- `mean_amt`, `amt_std` (historical behavior per card)
- `amt_vs_avg`, `amt_vs_avg_group`
- `z_score`
- `is_new_user`

I engineered these fields to make the analysis closer to real business use, not only basic descriptive statistics.

## 4) Business Metrics I Selected

In `notebooks/core_business_metrics.ipynb`, I focused on two levels of metrics:

1. **Global fraud KPIs**  
   (fraud rate and fraud loss rate for the whole dataset)
2. **Segment risk rates**  
   (fraud concentration by category, age group, city type, and distance behavior)

I selected these metrics because they are easy to explain to non-technical stakeholders and useful for risk prioritization.

## 5) Main Results

### Core KPI results
- Total transactions analyzed: **1,296,675**
- Fraudulent transactions: **7,506**
- Fraud transaction rate: **0.58%**
- Fraud loss rate: **4.4%** of total transaction amount

### Segment-level findings

**By transaction category (highest risk examples):**
- `shopping_net`: **1.76%**
- `misc_net`: **1.45%**
- `grocery_pos`: **1.41%**

**By age group (highest risk examples):**
- `65+`: **0.74%**
- `50-64`: **0.73%**
- `18-24`: **0.68%**

**By city type (highest risk examples):**
- `small city`: **0.68%**
- `medium city`: **0.65%**
- `large city`: **0.59%**

**By distance behavior (highest risk examples):**
- `Same Location (<1 km)`: **0.94%** (small volume segment)
- `City-Level (20-100 km)`: **0.58%**
- `Far (100-500 km)`: **0.58%**

From these results, I can see that fraud is not only a rare-event problem. Even with a low fraud count, the financial impact is important, and risk differs clearly across segments.

## 6) Machine Learning Results and Model Comparison

After finishing analytics, I trained and compared multiple models for fraud detection on imbalanced data.

I evaluated models with:
- ROC-AUC
- PR-AUC
- recall at high precision (I tracked best recall around precision 0.95)

### Logistic Regression (baseline)

Model:
- `LogisticRegression(C=0.01, class_weight="balanced", max_iter=1000, random_state=4)`

Results:
- ROC-AUC: **0.9238**
- PR-AUC: **0.2306**
- Best recall: **0.0000**

Conclusion:
- I used this as a baseline, but it was not good enough for fraud capture in this setup.

### Random Forest

Results:
- ROC-AUC: **0.9932**
- PR-AUC: **0.9247**
- Best recall: **0.8481**

Classification report highlights:
- Fraud class (`1`) precision: **0.98**
- Fraud class (`1`) recall: **0.73**
- Fraud class (`1`) F1-score: **0.83**

Conclusion:
- Random Forest performed much better than Logistic Regression and gave strong fraud detection performance.

### XGBoost (final best model)

Best tuned configuration from my search:
- `eta=0.10`
- `max_depth=9`
- `n_estimators=700`
- `gamma=0.1`
- `subsample=0.9`
- `min_child_weight=5`
- `colsample_bytree=0.9`
- `reg_alpha=0.3`

Best model validation metrics:
- ROC-AUC: **0.9993**
- PR-AUC: **0.9576**
- Best recall: **0.8521**

Test set metrics:
- ROC-AUC: **0.9983**
- PR-AUC (average precision): **0.9240**
- Best recall at 0.95 precision: **0.7883**

Conclusion:
- XGBoost gave my best overall results and is safer for large datasets, so this is my selected model for the project.

## 7) Project Files

- `notebooks/data_load.ipynb` - cleaning and engineered metrics
- `notebooks/core_business_metrics.ipynb` - business KPIs and risk analysis
- `notebooks/model_training.ipynb` - model training experiments
- `notebooks/tree_model.ipynb` and `notebooks/xgboost.ipynb` - tree-based modeling
- `notebooks/score_functions.py` - custom scoring functions

## 8) How to Reproduce My Analysis

1. Open the project in Jupyter Notebook or VS Code.
2. Run `notebooks/data_load.ipynb` to create the processed data file.
3. Run `notebooks/core_business_metrics.ipynb` to reproduce KPI and segment risk outputs.
4. Run `notebooks/model_training.ipynb`, `notebooks/tree_model.ipynb`, and `notebooks/xgboost.ipynb` to reproduce ML training and comparison.
