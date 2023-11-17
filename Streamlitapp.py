import streamlit as st
from keras.models import load_model
import numpy as np
import joblib

model = load_model('/Users/ericjr/Desktop/EricAfari_Assignment3/best_model.h5')

scaler = joblib.load('/Users/ericjr/Desktop/EricAfari_Assignment3/scaler_model.pkl')

st.title("Customer Churn Prediction")


def manual_encode(input_val, options):
    return options.index(input_val)


online_security_options = ['No', 'Yes']
tech_support_options = ['No', 'Yes']
contract_options = ['Month-to-month', 'One year', 'Two year']


online_security = st.selectbox("Online Security", online_security_options)

tech_support = st.selectbox("Tech Support", tech_support_options)

contract = st.selectbox("Contract", contract_options)


tenure = st.number_input("Tenure", min_value=0.0, max_value=100.0, value=0.0, step=0.1)


if st.button('Predict Churn'):
    features = [
        manual_encode(online_security, online_security_options),
        manual_encode(tech_support, tech_support_options),
        manual_encode(contract, contract_options),
        tenure
    ]

    features = np.array(features).reshape(1, -1)
    scaled_features = scaler.transform(features)

    prediction = model.predict(scaled_features)
    churn_probability = prediction[0][0]
    st.write(f'Churn Probability: {churn_probability:.2f}')
    if churn_probability > 0.5:
        st.write("Prediction: Churn (High Risk)")
        st.write(f"Confidence Level: {churn_probability:.2%}")
    else:
        st.write("Prediction: No Churn (Low Risk)")
        st.write(f"Confidence Level: {1 - churn_probability:.2%}")
model_accuracy = 0.7846481876332623

st.write(f"Model Accuracy (from testing): {model_accuracy:.2%}")
