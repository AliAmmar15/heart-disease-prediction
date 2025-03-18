import os
import numpy as np
import pandas as pd
import joblib
import yaml
import matplotlib.pyplot as plt  # ✅ Added import
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.inspection import PartialDependenceDisplay
from xgboost import XGBClassifier
from ucimlrepo import fetch_ucirepo
from utils import save_plot, fetch_cdc_data, fetch_fda_data, generate_report


model_dir = "models"
os.makedirs(model_dir, exist_ok=True)


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

cdc_df = fetch_cdc_data()
fda_df = fetch_fda_data()


heart_disease = fetch_ucirepo(id=45)
X = heart_disease.data.features
y = heart_disease.data.targets.map(lambda x: 1 if x in {1, 2, 3} else 0).values.ravel()


imputer = SimpleImputer(strategy="mean")
X.loc[:, ['ca', 'thal']] = imputer.fit_transform(X[['ca', 'thal']])


if cdc_df is not None and len(cdc_df) > 0:
    X = pd.concat([X.reset_index(drop=True), cdc_df.reset_index(drop=True)], axis=1)

if fda_df is not None and len(fda_df) > 0:
    X = pd.concat([X.reset_index(drop=True), fda_df.reset_index(drop=True)], axis=1)


X = X.apply(pd.to_numeric, errors='coerce').fillna(0)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=config["data"]["test_size"], random_state=config["data"]["random_state"], stratify=y)


rf_param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 20, None],
    "min_samples_split": [2, 5, 10]
}
rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=5, n_jobs=-1, verbose=2)
rf_grid_search.fit(X_train, y_train)
best_rf = rf_grid_search.best_estimator_


xgb_model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
xgb_model.fit(X_train, y_train)


joblib.dump(best_rf, os.path.join(model_dir, "random_forest_model.pkl"))
joblib.dump(xgb_model, os.path.join(model_dir, "xgboost_model.pkl"))

print("\n Models saved successfully in 'models/' folder!")


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Best Random Forest": best_rf,
    "XGBoost": xgb_model
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    
    report = classification_report(y_test, y_pred, zero_division=1)


    with open(f"results/{name.replace(' ', '_')}_report.md", "w") as f:
        f.write(generate_report(name, report))
    
    print(f"\n{name} Classification Report:\n{report}")

   
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Blues", ax=ax)
    plt.title(f"Confusion Matrix: {name}")
    save_plot(fig, f"{name.replace(' ', '_')}_confusion_matrix.png")

print("\n Training complete! Models saved in 'models/' folder. Results saved in 'results/'!")
