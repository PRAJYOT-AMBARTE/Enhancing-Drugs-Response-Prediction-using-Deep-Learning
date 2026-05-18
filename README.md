# Enhancing-Drugs-Response-Prediction-using-Deep-Learning
Developed a deep learning-based predictive model to analyze patient and molecular data for accurate drug response forecasting, improving precision medicine and treatment optimization.

## Drug Response Prediction - Simplified Demo

This project predicts drug response for a synthetic disease using 3 ML models.
The best model (NeuralNet) is used in a Streamlit GUI.

### How to Run

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Launch GUI:
   ```bash
   streamlit run gui/app.py
   ```

### Files
- `data/synthetic_drug_response.csv` - synthetic dataset
- `gui/neural_net_model.pkl` - trained model
- `gui/app.py` - Streamlit app

## Future Improvements

py model -- host -- api
api integration -- frontend 
react frontend -- vercel 

frontend (minimum) __
   auth(login , signup)
   profile
   dashboard
   index page-- calculation happens here (api calling)

model--
    <!-- if prediction == 1:
        st.success(f"✅ Predicted Positive Response to {drug_selected} (Confidence: {prob:.2f})")
    else:
        st.error(f"❌ Predicted No Response to {drug_selected} (Confidence: {1 - prob:.2f})") -->

proper message (with charts / statistics)
