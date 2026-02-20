import os
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from .config import EXPERIMENT_NAME, TARGET_COLUMN, RANDOM_STATE
from .preprocessing import build_tree_preprocessor
from .models import get_xgb_model

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(EXPERIMENT_NAME)


def train_calibrated_xgb():

    # -----------------------
    # Load Data
    # -----------------------
    df = pd.read_csv("data/raw/train.csv")
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Absence": 0, "Presence": 1})

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # -----------------------
    # Build Pipeline
    # -----------------------
    preprocessor = build_tree_preprocessor()
    base_model = get_xgb_model()

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", base_model)
    ])

    calibrated_model = CalibratedClassifierCV(
        pipeline,
        method="sigmoid",
        cv=5
    )

    calibrated_model.fit(X_train, y_train)

    # -----------------------
    # Validation Predictions
    # -----------------------
    val_probs = calibrated_model.predict_proba(X_val)[:, 1]
    val_preds = (val_probs >= 0.5).astype(int)

    # -----------------------
    # Core Metrics
    # -----------------------
    roc_auc = roc_auc_score(y_val, val_probs)
    pr_auc = average_precision_score(y_val, val_probs)
    accuracy = accuracy_score(y_val, val_preds)
    precision = precision_score(y_val, val_preds)
    recall = recall_score(y_val, val_preds)
    f1 = f1_score(y_val, val_preds)

    tn, fp, fn, tp = confusion_matrix(y_val, val_preds).ravel()

    # -----------------------
    # Threshold Sweep
    # -----------------------
    thresholds = np.arange(0.1, 0.95, 0.05)
    best_f1 = 0
    best_threshold = 0.5

    for t in thresholds:
        preds_t = (val_probs >= t).astype(int)
        f1_t = f1_score(y_val, preds_t)
        if f1_t > best_f1:
            best_f1 = f1_t
            best_threshold = t

    # -----------------------
    # Classification Report
    # -----------------------
    report = classification_report(y_val, val_preds)

    os.makedirs("data/artifacts", exist_ok=True)
    report_path = "data/artifacts/xgb_calibrated_classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    # -----------------------
    # Calibration Curve Plot
    # -----------------------
    prob_true, prob_pred = calibration_curve(y_val, val_probs, n_bins=10)

    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("Calibration Curve")
    plt.xlabel("Predicted Probability")
    plt.ylabel("True Probability")

    calib_path = "data/artifacts/calibration_curve.png"
    plt.savefig(calib_path)
    plt.close()

    # -----------------------
    # Save Joblib
    # -----------------------
    os.makedirs("models", exist_ok=True)
    joblib_path = "models/xgb_calibrated.joblib"
    joblib.dump(calibrated_model, joblib_path)

    # -----------------------
    # Log To MLflow
    # -----------------------
    with mlflow.start_run(run_name="XGBoost_Calibrated_FullMetrics"):

        mlflow.log_params({
            "model": "XGBoost_Calibrated",
            "calibration_method": "sigmoid",
            "threshold_default": 0.5
        })

        mlflow.log_metrics({
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "accuracy_0.5": accuracy,
            "precision_0.5": precision,
            "recall_0.5": recall,
            "f1_0.5": f1,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
            "best_f1": best_f1
        })

        mlflow.log_param("best_threshold", best_threshold)

        mlflow.log_artifact(report_path)
        mlflow.log_artifact(calib_path)
        mlflow.log_artifact(joblib_path)

        mlflow.sklearn.log_model(
            calibrated_model,
            artifact_path="model",
            registered_model_name="Heart_Disease_XGB_Calibrated"
        )

    print("Full metrics logged successfully.")
    print("Validation ROC-AUC:", roc_auc)