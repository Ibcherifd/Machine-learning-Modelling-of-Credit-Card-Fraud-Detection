import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('xgboost_model.pkl')

# Page title
st.title("Credit Card Fraud Detection")

# Input form
st.write("Enter PCA-transformed features for prediction:")
features = []
for i in range(30):  # assuming 30 features
    val = st.number_input(f"Feature {i+1}", value=0.0)
    features.append(val)

# Predict button
if st.button("Predict"):
    prediction = model.predict([features])
    prob = model.predict_proba([features])[0][1]
    if prediction[0] == 1:
        st.error(f"⚠️ Fraud Detected! (Confidence: {prob:.2f})")
    else:
        st.success(f"This is Legitimate")
