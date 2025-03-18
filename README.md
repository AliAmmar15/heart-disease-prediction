# Heart Disease Prediction Project

This is a simple machine learning project designed to predict the presence of heart disease using patient health data. It leverages **Logistic Regression**, **Random Forest**, and **XGBoost** models to determine if patients are at risk.

## 📂 Project Structure

```
.
├── models/
│   ├── random_forest_model.pkl
│   └── xgboost_model.pkl
├── src/
│   ├── ml_model_analysis.py
│   ├── predict.py
│   └── utils.py
├── results/
│   ├── confusion_matrix.png
│   └── feature_importance.png
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/AliAmmar15/heart-disease-prediction.git
cd heart-disease-prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Models

Run the script to train and save your models:

```bash
python src/ml_model_analysis.py
```

Your trained models will be saved in the `models/` folder. Visualizations and evaluation metrics will be saved in the `results/` folder.

### 4. Make Predictions

To predict heart disease risk, run:

```bash
python src/predict.py
```

The output will clearly indicate predictions for each patient, specifying whether the patient is at risk or not.

## 🛠 Requirements

- Python (3.8 or newer)
- Dependencies listed in `requirements.txt`

## 📊 Dataset

This project uses the [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease).

## 📌 Results

Classification reports, confusion matrix, and feature importance visualizations are automatically generated and saved in the `results/` folder.

---

**Happy coding! 🚑📈**