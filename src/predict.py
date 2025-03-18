import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
from utils import fetch_cdc_data, fetch_fda_data


model_dir = "models"
model_path_rf = os.path.join(model_dir, "random_forest_model.pkl")
model_path_xgb = os.path.join(model_dir, "xgboost_model.pkl")


if not os.path.exists(model_path_rf) or not os.path.exists(model_path_xgb):
    raise FileNotFoundError(" Model files not found! Please run 'python src/ml_model_analysis.py' first.")


rf_model = joblib.load(model_path_rf)
xgb_model = joblib.load(model_path_xgb)

print(" Models loaded successfully from 'models/' folder!")


heart_disease = fetch_ucirepo(id=45)
X = heart_disease.data.features


cdc_df = fetch_cdc_data()
fda_df = fetch_fda_data()


X = X.apply(pd.to_numeric, errors='coerce').fillna(0)


if cdc_df is not None and len(cdc_df) > 0:
    X = pd.concat([X.reset_index(drop=True), cdc_df.reset_index(drop=True)], axis=1)

if fda_df is not None and len(fda_df) > 0:
    X = pd.concat([X.reset_index(drop=True), fda_df.reset_index(drop=True)], axis=1)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_test = X_scaled[-5:]


prediction_labels = {0: "No Disease", 1: "Has Disease"}


rf_predictions = rf_model.predict(X_test)
xgb_predictions = xgb_model.predict(X_test)


print("\n🔹 **Model Predictions:**\n")

print(" **Using Random Forest:**")
for i, pred in enumerate(rf_predictions, start=1):
    print(f"   - Patient {i}: {prediction_labels[pred]}")

print("\n **Using XGBoost:**")
for i, pred in enumerate(xgb_predictions, start=1):
    print(f"   - Patient {i}: {prediction_labels[pred]}")
