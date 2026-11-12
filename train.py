"""Train the heart disease classifier and save it to model/heart_model.joblib.

Usage:
    pip install -r requirements.txt kagglehub
    python train.py                      # downloads dataset via kagglehub
    python train.py --csv data/heart.csv # or use a local CSV
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
OUT = Path("model/heart_model.joblib")


def load_data(csv_path):
    if csv_path:
        return pd.read_csv(csv_path)
    import kagglehub
    folder = Path(kagglehub.dataset_download("johnsmith88/heart-disease-dataset"))
    return pd.read_csv(next(folder.glob("*.csv")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="Path to local heart.csv")
    args = parser.parse_args()

    df = load_data(args.csv).drop_duplicates()
    X, y = df[FEATURES], df["target"]  # target: 1 = disease, 0 = no disease
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
    ])
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")
    print(classification_report(y_test, preds))

    OUT.parent.mkdir(exist_ok=True)
    joblib.dump(model, OUT)
    print(f"Saved model to {OUT}")


if __name__ == "__main__":
    main()
