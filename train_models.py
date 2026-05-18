import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib
import os

# Set up folders
os.makedirs("data", exist_ok=True)
os.makedirs("gui", exist_ok=True)

# Generate synthetic data
np.random.seed(42)
num_patients = 100
num_features = 10
num_drugs = 5
patient_features = np.random.rand(num_patients, num_features)

drug_data = {}
for i in range(num_drugs):
    weights = np.random.rand(num_features)
    scores = patient_features @ weights + np.random.normal(0, 0.1, num_patients)
    threshold = np.median(scores)
    response = (scores > threshold).astype(int)
    drug_data[f"Drug_{i+1}"] = response

# Create dataframe
columns = [f"feature_{i+1}" for i in range(num_features)]
df = pd.DataFrame(patient_features, columns=columns)
for drug, response in drug_data.items():
    df[drug] = response

# Save dataset
df.to_csv("data/synthetic_drug_response.csv", index=False)

# Use Drug_1 for training and evaluation
X = df[[col for col in df.columns if col.startswith("feature_")]]
y = df["Drug_1"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True, kernel='rbf', random_state=42),
    "NeuralNet": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
}

# Train and evaluate
best_model = None
best_auc = 0
print("Model Evaluation Results:\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print(f"{name}: Accuracy={acc:.2f}, F1={f1:.2f}, AUC={auc:.2f}")
    
    if auc > best_auc:
        best_auc = auc
        best_model = model

# Save the best model
joblib.dump(best_model, "gui/neural_net_model.pkl")
print("\nBest model saved as 'gui/neural_net_model.pkl'")