# Customer Churn Prediction

End-to-end Telco customer churn classification with preprocessing, model artifacts, SHAP explainability, and a Streamlit prediction app.

## Structure

- `notebooks/customer_prediction.ipynb` — EDA, preprocessing, training, evaluation, and explainability.
- `data/` — local Telco churn dataset.
- `src/app.py` — Streamlit app for live churn prediction.
- `artifacts/models/` — trained model and preprocessing pipeline.
- `artifacts/plots/` — generated evaluation and SHAP figures.
- `requirements.txt` — project dependencies.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run src/app.py
```

