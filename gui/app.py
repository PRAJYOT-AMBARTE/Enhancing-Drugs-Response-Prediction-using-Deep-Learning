
import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load model (trained for all 5 drugs using same feature ordering)
model_paths = {
    "Metformin": "gui/neural_net_model.pkl",
    "Dapagliflozin": "gui/neural_net_model.pkl",
    "Sitagliptin": "gui/neural_net_model.pkl",
    "Empagliflozin": "gui/neural_net_model.pkl",
    "Glipizide": "gui/neural_net_model.pkl"
}

st.title("Drug Response Prediction for Type 2 Diabetes")
st.markdown("Enter patient clinical features to predict if a selected drug will be effective.")

# Dropdown to select the drug
drug_selected = st.selectbox(
    "Select Drug",
    ["Metformin", "Dapagliflozin", "Sitagliptin", "Empagliflozin", "Glipizide"]
)

# Display friendly feature names
feature_names = [
    "Age",
    "BMI",
    "Blood Pressure",
    "Glucose Level",
    "Insulin Sensitivity",
    "Cholesterol Level",
    "Physical Activity Score",
    "Smoking Index",
    "Alcohol Consumption Score",
    "Genetic Risk Score"
]

# Collect inputs using sliders
st.subheader("Patient Features")
feature_inputs = []
for name in feature_names:
    val = st.slider(name, min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    feature_inputs.append(val)

# Predict using model
if st.button("Predict Drug Response"):
    model = joblib.load(model_paths[drug_selected])
    prediction = model.predict([feature_inputs])[0]
    prob = model.predict_proba([feature_inputs])[0][1]

    if prediction == 1:
        st.success(f"✅ Predicted Positive Response to {drug_selected} (Confidence: {prob:.2f})")
    else:
        st.error(f"❌ Predicted No Response to {drug_selected} (Confidence: {1 - prob:.2f})")
