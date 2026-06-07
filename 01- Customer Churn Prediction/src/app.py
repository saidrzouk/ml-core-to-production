from pathlib import Path

import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]
model_path = ROOT_DIR / 'artifacts' / 'models' / 'model.pkl'
processor_path = ROOT_DIR / 'artifacts' / 'models' / 'preprocessor.pkl'

model = joblib.load(model_path)
preprocessor = joblib.load(processor_path)

st.title('Customer Churn Predictor')
st.markdown(
    'Estimate churn risk for a customer profile using a trained XGBoost model. '
    'This app uses the same preprocessing pipeline as the analysis notebook.'
)

col1, col2 = st.columns(2)
with col1:
    tenure = st.slider('Tenure (months)', 0, 72, 12)
    monthly = st.number_input('Monthly charges ($)', 20.0, 120.0, 65.0)
    contract = st.selectbox(
        'Contract type', ['Month-to-month', 'One year', 'Two year']
    )
    internet = st.selectbox('Internet service', ['DSL', 'Fiber optic', 'No'])
with col2:
    tech_support   = st.selectbox('Tech support', ['Yes','No','No internet service'])
    payment        = st.selectbox('Payment method', [
        'Electronic check','Mailed check',
        'Bank transfer (automatic)','Credit card (automatic)'])
    senior         = st.selectbox('Senior citizen', ['Yes','No'])
    partner        = st.selectbox('Has partner', ['Yes','No'])

if st.button('Predict churn'):
    input_df = pd.DataFrame([{
        'tenure': tenure, 'MonthlyCharges': monthly,
        'TotalCharges': tenure * monthly,
        'Contract': contract, 'InternetService': internet,
        'TechSupport': tech_support, 'PaymentMethod': payment,
        'SeniorCitizen': 1 if senior=='Yes' else 0,
        'Partner': partner,
        'gender':'Male','Dependents':'No','PhoneService':'Yes',
        'MultipleLines':'No','OnlineSecurity':'No','OnlineBackup':'No',
        'DeviceProtection':'No','StreamingTV':'No','StreamingMovies':'No',
        'PaperlessBilling':'Yes'
    }])
    X_proc = preprocessor.transform(input_df)
    proba  = model.predict_proba(X_proc)[0][1]

    st.metric('Churn probability', f'{proba*100:.1f}%')
    if proba > 0.5:
        st.error('High churn risk')
    else:
        st.success('Low churn risk')

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_proc)
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_proc, show=False, plot_type='bar')
    st.pyplot(fig)