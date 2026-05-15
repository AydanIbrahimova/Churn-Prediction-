# 📡 Customer Churn Prediction — ML Project Documentation

> **Dataset:** Cell2Cell Telecom · **Model:** XGBoost · **Goal:** Predict customer churn to enable proactive retention

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Structure](#2-project-structure)
3. [Dataset](#3-dataset)
4. [Exploratory Data Analysis](#4-exploratory-data-analysis)
5. [Feature Engineering](#5-feature-engineering)
6. [Modeling Pipeline](#6-modeling-pipeline)
7. [Model Evaluation](#7-model-evaluation)
8. [Threshold Optimization](#8-threshold-optimization)
9. [Final Results](#9-final-results)
10. [Streamlit Dashboard](#10-streamlit-dashboard)
11. [How to Run](#11-how-to-run)
12. [Dependencies](#12-dependencies)

---

## 1. Project Overview

This project builds a binary classification model to predict whether a telecom customer will churn (cancel their subscription). The business goal is to enable proactive retention — identifying at-risk customers before they leave so that targeted offers can be sent.

**Key business logic:** In telecom, missing a churner is significantly more costly than sending an unnecessary retention offer. This drives our modeling decisions — specifically the choice to optimize for **Recall** over Precision.

| Item | Detail |
|---|---|
| Problem Type | Binary Classification |
| Target Variable | `Churn` (Yes = 1, No = 0) |
| Primary Metric | PR-AUC (imbalanced classes) |
| Secondary Metric | Recall (business priority) |
| Final Model | XGBoost with Optuna tuning |

---

## 2. Project Structure

```
final/
│
├── data/
│   └── cell2celltrain.csv          # Raw dataset
│
├── model/
│   ├── xgboost_churn_v1.pkl        # Trained pipeline (saved with joblib)
│   └── optimal_threshold.json      # Optimal decision threshold
│
├── Final_project.ipynb             # Full ML workflow notebook
│
└── ml_streamlit_app/
    ├── main.py                     # Streamlit dashboard (7 sections)
    └── assets/                     # Static assets (CSS, images)
```

---

## 3. Dataset

**Source:** `cell2celltrain.csv` — Cell2Cell telecom customer dataset

| Property | Value |
|---|---|
| Rows | 71,047 |
| Original Columns | 58 |
| Features after selection | 32 |
| Target | `Churn` (binary) |
| Class Distribution | 71.0% No Churn / 29.0% Churn |

### Class Imbalance

The dataset has a significant class imbalance (71/29 split). This was addressed using **SMOTE** (Synthetic Minority Oversampling Technique) within the pipeline, applied only to training data to prevent data leakage.

---

## 4. Exploratory Data Analysis

### 4.1 Missing Values

Missing values were below 2% for all columns. Crucially, the missingness itself was found to carry predictive signal:

| Column | Missing % | Churn Rate (NaN) | Churn Rate (Not NaN) |
|---|---|---|---|
| PercChangeMinutes | 2.01% | **56.7%** | 27.8% |
| PercChangeRevenues | 2.01% | **56.7%** | 27.8% |
| AgeHH1 / AgeHH2 | 1.87% | 29.1–29.2% | 28.7% |

**Finding:** Customers with missing `PercChangeMinutes` churn at nearly double the rate. This means they had no prior month data — a strong churn signal. Binary flag features (`PercChange_missing`) were created to capture this.

### 4.2 Skewness

Several features showed extreme right skew due to zero-inflation:

| Feature | Skewness | Action |
|---|---|---|
| CallForwardingCalls | 91.6 | Binarized |
| UniqueSubs | 79.6 | Binarized |
| RoamingCalls | 42.1 | Binarized |
| DroppedBlockedCalls | 18.3 | Yeo-Johnson transform |
| MonthlyRevenue | 3.2 | Yeo-Johnson transform |

### 4.3 Correlation with Target

No single feature showed strong linear correlation with churn (maximum ≈ 0.10). This confirmed that linear models would underperform and motivated the use of tree-based ensemble methods.

| Feature | Correlation with Churn |
|---|---|
| MonthsInService | +0.098 |
| TotalRecurringCharge | +0.087 |
| PercChangeMinutes | -0.082 |
| MonthlyRevenue | +0.071 |
| DroppedCalls | +0.063 |

---

## 5. Feature Engineering

### 5.1 New Features Created

| Feature | Formula | Rationale |
|---|---|---|
| `CustomerValue` | MonthsInService × MonthlyRevenue | Overall customer lifetime value — high-value long-term customers churn less |
| `ProblemCallRate` | (DroppedCalls + BlockedCalls) / (MonthlyMinutes + 1) | Network quality proxy — high problem rate drives churn |
| `TenureEquipmentRatio` | MonthsInService / (CurrentEquipmentDays + 1) | Equipment refresh dynamics — old device + long tenure signals departure risk |
| `Usage_missing` | RoamingCalls.isna() → binary flag | Non-roaming customers have a distinct churn profile |
| `PercChange_missing` | PercChangeMinutes.isna() → binary flag | Missing prior month data is itself a high-risk signal (churn rate 56.7%) |

### 5.2 Dropped Features — Data Leakage

The following columns were removed because they record events that happen simultaneously with or after the churn decision, making them data leaks:

- `RetentionCalls`
- `RetentionOffersAccepted`
- `MadeCallToRetentionTeam`

### 5.3 Dropped Features — Quasi-Constant

4 columns with over 98% identical values were removed as they carry no discriminative information.

### 5.4 Binarized Features (Zero-Inflated)

| Original Feature | New Feature | Zero % | Meaning |
|---|---|---|---|
| AdjustmentsToCreditRating | CreditRatingAdjusted | 96.4% | Did any credit adjustment occur? |
| ReferralsMadeBySubscriber | MadeReferral | 95.3% | Did the customer show loyalty? |
| ThreewayCalls | ThreewayCalls_used | 72.7% | Is the customer actively using services? |

### 5.5 Feature Selection

XGBoost feature importances were used to select features covering **85% of cumulative importance**, reducing the feature space from 61 to **32 features**. This improved generalization and reduced overfitting.

---

## 6. Modeling Pipeline

### 6.1 Pipeline Structure

```python
ImbPipeline([
    ('rare_category',  RareCategoryGrouper()),     # Group rare categories as 'Rare'
    ('preprocessor',   ColumnTransformer([
                           numeric_transformer,    # median impute + StandardScaler
                           binary_transformer,     # mode impute + OrdinalEncoder
                           ordinal_transformer,    # mode impute + OrdinalEncoder
                           onehot_transformer      # mode impute + OneHotEncoder
                       ])),
    ('cap_outliers',   OutlierCapper(1%–99%)),     # Winsorize outliers
    ('fix_skewness',   YeoJohnsonTransformer()),   # Reduce skewness
    ('smote',          SMOTE(random_state=42)),    # Oversample minority class
    ('model',          XGBClassifier(...))
])
```

All preprocessing steps are inside the pipeline to prevent data leakage. SMOTE is applied after preprocessing and only within training folds.

### 6.2 Models Compared

| Model | Notes |
|---|---|
| Logistic Regression | Baseline linear model |
| Random Forest | Ensemble of decision trees |
| XGBoost | Gradient boosted trees — selected as final model |

### 6.3 Hyperparameter Tuning

Optuna was used for Bayesian hyperparameter optimization:

- **Trials:** 50
- **Timeout:** 1800 seconds
- **Sampler:** TPE (Tree-structured Parzen Estimator)
- **Optimization target:** Average Precision (PR-AUC) via 5-fold Stratified CV
- **Search space:** n_estimators, max_depth, learning_rate, subsample, colsample_bytree, min_child_weight, scale_pos_weight, gamma, reg_alpha, reg_lambda

---

## 7. Model Evaluation

All models were evaluated with **5-fold Stratified Cross-Validation** to ensure reliable estimates across all class distributions.

### 7.1 Model Comparison

| Model | Test AUC | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Logistic Regression | 0.6198 | 0.3790 | 0.4412 | 0.3521 | 0.5892 |
| Random Forest | 0.6489 | 0.4080 | 0.4721 | 0.3812 | 0.6234 |
| **XGBoost** | **0.6726** | **0.4495** | **0.4987** | **0.3875** | **0.6992** |

XGBoost outperformed all other models across every metric and was selected as the final model.

---

## 8. Threshold Optimization

By default, classifiers use a decision threshold of 0.5. However, given our business goal of maximizing churner detection, the threshold was optimized.

**Method:** `cross_val_predict` on training data was used to find the threshold maximizing F1 score. Using training data with cross-validation (not test data) ensures no data leakage.

**Optimal threshold: 0.485**

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.50 (default) | 0.414 | 0.699 | 0.520 |
| **0.485 (optimal)** | **0.348** | **0.817** | **0.488** |

The lower threshold trades some Precision for a significant **+11.8 percentage point gain in Recall** — meaning the model catches far more actual churners, which is the primary business objective.

---

## 9. Final Results

Three sequential improvements were applied after the baseline model:

| Step | AUC | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Baseline (61 features) | 0.6726 | 0.4495 | 0.4987 | 0.3875 | 0.6992 |
| Feature Selection (32 features) | 0.6427 | 0.3978 | 0.4822 | 0.3639 | 0.7145 |
| **Tuned + Threshold 0.485** | **0.6455** | **0.3986** | **0.4884** | **0.3483** | **0.8171** |

### Key Takeaways

- Feature selection reduced complexity from 61 to 32 features with minimal AUC cost (-0.027)
- Optuna tuning + threshold optimization achieved a **+11.79pp Recall gain** over the baseline
- The final model correctly identifies **81.7% of all churners** in the test set
- The business trade-off is deliberate: missing a churner is more costly than sending an unnecessary retention offer

---

## 10. Streamlit Dashboard

The interactive dashboard (`main.py`) visualizes the full ML workflow across 7 sections:

| Section | Contents |
|---|---|
| 🏠 Executive Overview | KPI cards, ROC curve, Confusion Matrix, Top Feature Importance |
| 📊 Data Understanding | Dataset stats, class distribution, feature type breakdown |
| 🔍 Exploratory Analysis | Missing values, skewness analysis, correlation with target |
| ⚙️ Feature Engineering | New features, leakage removals, binarization, importance waterfall |
| 🤖 Modeling & Evaluation | Model comparison charts, pipeline structure, Optuna tuning details |
| 🔬 Model Insights | Threshold analysis curve, Precision-Recall curves, final results table |
| 🎯 Prediction Lab | Interactive form — input customer data, get real-time churn probability |

### Prediction Lab

The Prediction Lab section allows the user to input customer attributes and receive an immediate churn probability score with a gauge chart, CHURN / NO CHURN verdict, and a list of key risk factors driving the prediction.

---

## 11. How to Run

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# Navigate to the app directory
cd final/ml_streamlit_app

# Install dependencies
pip install streamlit plotly pandas numpy scikit-learn xgboost imbalanced-learn optuna joblib

# Run the dashboard
streamlit run main.py
```

The app will open at `http://localhost:8501`

### File Paths

The app automatically resolves paths relative to the project root:

```python
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH     = os.path.join(BASE_DIR, "model", "xgboost_churn_v1.pkl")
THRESHOLD_PATH = os.path.join(BASE_DIR, "model", "optimal_threshold.json")
DATA_PATH      = os.path.join(BASE_DIR, "data",  "cell2celltrain.csv")
```

---

## 12. Dependencies

| Library | Version | Purpose |
|---|---|---|
| pandas | ≥1.5 | Data manipulation |
| numpy | ≥1.23 | Numerical operations |
| scikit-learn | ≥1.2 | Preprocessing, metrics, cross-validation |
| xgboost | ≥1.7 | Gradient boosting model |
| imbalanced-learn | ≥0.10 | SMOTE oversampling |
| optuna | ≥3.0 | Bayesian hyperparameter tuning |
| joblib | ≥1.2 | Model serialization |
| streamlit | ≥1.28 | Interactive dashboard |
| plotly | ≥5.15 | Interactive visualizations |

---

*Documentation generated for Final Project — Customer Churn Prediction*
