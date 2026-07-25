"""
modelling.py (MLflow Project entry point)
=========================================
Script training untuk CI. Dijalankan melalui `mlflow run` sehingga menerima
parameter dari file MLProject. Model dilatih dan dicatat ke MLflow tracking
lokal (folder mlruns), yang kemudian diunggah sebagai artefak oleh GitHub
Actions.
"""

import argparse
import os

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

TARGET = "Churn"


def load_data(data_path):
    train = pd.read_csv(os.path.join(data_path, "train.csv"))
    test = pd.read_csv(os.path.join(data_path, "test.csv"))
    X_train, y_train = train.drop(columns=[TARGET]), train[TARGET]
    X_test, y_test = test.drop(columns=[TARGET]), test[TARGET]
    return X_train, X_test, y_train, y_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--data_path", type=str, default="telco_preprocessing")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_data(args.data_path)

    # autolog + manual logging metrik evaluasi
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="CI_RandomForest"):
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("test_precision", precision_score(y_test, y_pred))
        mlflow.log_metric("test_recall", recall_score(y_test, y_pred))
        mlflow.log_metric("test_f1", f1_score(y_test, y_pred))

        print("[OK] Model CI selesai dilatih.")
        print(f"     n_estimators={args.n_estimators}, max_depth={args.max_depth}")
        print(f"     Test accuracy: {accuracy_score(y_test, y_pred):.4f}")


if __name__ == "__main__":
    main()
