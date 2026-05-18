
import streamlit as st
import numpy as np
import joblib

# Load model
model_paths = {
    "Metformin": "gui/neural_net_model.pkl",
    "Dapagliflozin": "gui/neural_net_model.pkl",
    "Sitagliptin": "gui/neural_net_model.pkl",
    "Empagliflozin": "gui/neural_net_model.pkl",
    "Glipizide": "gui/neural_net_model.pkl"
}

# Feature display and normalization ranges
feature_display = [
    ("Age", 18, 80),
    ("BMI", 15, 40),
    ("Blood Pressure", 80, 180),
    ("Glucose Level", 70, 200),
    ("Insulin Sensitivity", 0.5, 5.0),
    ("Cholesterol Level", 100, 300),
    ("Physical Activity Score", 0, 10),
    ("Smoking Index", 0, 10),
    ("Alcohol Consumption Score", 0, 10),
    ("Genetic Risk Score", 0, 1)
]

st.title("Drug Response Prediction for Type 2 Diabetes")
st.markdown("Enter real patient values to predict if a selected drug will be effective.")

drug_selected = st.selectbox(
    "Select Drug",
    list(model_paths.keys())
)

# Slider with real values
st.subheader("Patient Features")
normalized_inputs = []
for name, min_val, max_val in feature_display:
    val = st.slider(f"{name} ({min_val}-{max_val})", float(min_val), float(max_val), float((min_val + max_val)/2), step=0.1)
    norm_val = (val - min_val) / (max_val - min_val)
    normalized_inputs.append(norm_val)

if st.button("Predict Drug Response"):
    model = joblib.load(model_paths[drug_selected])
    prediction = model.predict([normalized_inputs])[0]
    prob = model.predict_proba([normalized_inputs])[0][1]

    if prediction == 1:
        st.success(f"✅ Predicted Positive Response to {drug_selected} (Confidence: {prob:.2f})")
    else:
        st.error(f"❌ Predicted No Response to {drug_selected} (Confidence: {1 - prob:.2f})")
