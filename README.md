# Drug Response Prediction - Simplified Demo

This project predicts drug response for a synthetic disease using 3 ML models.
The best model (NeuralNet) is used in a Streamlit GUI.

## How to Run

1. Install requirements:
   ```
   pip install -r requirements.txt
   ```

2. Launch GUI:
   ```
   streamlit run gui/app.py
   ```

## Files
- `data/synthetic_drug_response.csv` - synthetic dataset
- `gui/neural_net_model.pkl` - trained model
- `gui/app.py` - Streamlit app



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