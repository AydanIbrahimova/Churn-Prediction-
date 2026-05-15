# Churn-Prediction-
End-to-end ML project predicting telecom customer churn on 71K records. XGBoost pipeline with custom feature engineering, SMOTE, Optuna tuning (50 trials), and threshold optimization achieving 81.7% Recall. Includes a 7-section interactive Streamlit dashboard.
# 📡 Customer Churn Prediction

Binary classification model to predict telecom customer churn using the Cell2Cell dataset (71,047 records).

## Results

| Metric | Score |
|--------|-------|
| AUC-ROC | 0.6455 |
| Recall | **81.7%** |
| F1 | 0.488 |
| Threshold | 0.485 |

## Stack

- **Model:** XGBoost with Optuna hyperparameter tuning (50 trials, TPE Sampler)
- **Imbalance:** SMOTE oversampling
- **Dashboard:** Streamlit (7 sections)

## Project Structure

```
final/
├── data/                        # Dataset (not included)
├── model/                       # Trained model + threshold
├── Final_project.ipynb          # Full ML workflow
└── ml_streamlit_app/
    └── main.py                  # Streamlit dashboard
```

## Run

```bash
cd ml_streamlit_app
pip install streamlit plotly pandas numpy scikit-learn xgboost imbalanced-learn
streamlit run main.py
```

## Dataset

[Cell2Cell Telecom Churn Dataset](https://www.kaggle.com/datasets/jpacse/datasets-for-churn-telecom)
