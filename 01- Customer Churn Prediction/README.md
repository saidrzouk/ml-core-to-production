# Customer Churn Prediction

End-to-end Telco customer churn classification with preprocessing, model artifacts, SHAP explainability, and a Streamlit prediction app.

## Structure

- `notebooks/customer_prediction.ipynb` — EDA, preprocessing, training, evaluation, and explainability.
- `data/` — local Telco churn dataset.
- `src/config.py` — environment-configurable paths for model and preprocessor artifacts.
- `src/app.py` — Streamlit app for live churn prediction.
- `artifacts/models/` — trained model and preprocessing pipeline.
- `artifacts/plots/` — generated evaluation and SHAP figures.
- `requirements.txt` — project dependencies.

## Configuration

This project supports overriding artifact paths with environment variables:

- `CHURN_ARTIFACT_DIR` — path to the artifact directory.
- `CHURN_MODEL_PATH` — path to the model file.
- `CHURN_PREPROCESSOR_PATH` — path to the preprocessing pipeline.

If unset, paths default to `artifacts/models/model.pkl` and `artifacts/models/preprocessor.pkl`.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run src/app.py
```

