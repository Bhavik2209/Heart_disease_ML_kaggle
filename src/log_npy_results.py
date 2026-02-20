import os
import numpy as np
import pandas as pd
import mlflow
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)
from config import EXPERIMENT_NAME

# --- Force same tracking backend ---
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(EXPERIMENT_NAME)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
NPY_DIR = os.path.join(BASE_DIR, "npy_results")
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "train.csv")

# --- Load ground truth ---
df = pd.read_csv(DATA_PATH)
df["Heart Disease"] = df["Heart Disease"].map({"Absence": 0, "Presence": 1})
y = df["Heart Disease"].values

# --- Load OOF predictions ---
oof_files = {
    "LightGBM": "oof_lgb.npy",
    "XGBoost": "oof_xgb.npy",
    "Blend": "oof_blend.npy",
    "Stacked": "oof_stacked.npy"
}

def log_metrics(name, preds):

    with mlflow.start_run(run_name=f"{name}_From_NPY"):

        # ---- ROC + PR AUC ----
        roc_auc = roc_auc_score(y, preds)
        pr_auc = average_precision_score(y, preds)

        # ---- Default threshold (0.5) ----
        binary_preds = (preds >= 0.5).astype(int)

        acc = accuracy_score(y, binary_preds)
        precision = precision_score(y, binary_preds)
        recall = recall_score(y, binary_preds)
        f1 = f1_score(y, binary_preds)

        tn, fp, fn, tp = confusion_matrix(y, binary_preds).ravel()

        # ---- Log Core Metrics ----
        mlflow.log_metrics({
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "accuracy_0.5": acc,
            "precision_0.5": precision,
            "recall_0.5": recall,
            "f1_0.5": f1,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp
        })

        # ---- Threshold Sweep ----
        thresholds = np.arange(0.1, 0.95, 0.05)
        best_f1 = 0
        best_threshold = 0.5

        for t in thresholds:
            preds_t = (preds >= t).astype(int)
            f1_t = f1_score(y, preds_t)

            mlflow.log_metric(f"f1_t_{round(t,2)}", f1_t)

            if f1_t > best_f1:
                best_f1 = f1_t
                best_threshold = t

        mlflow.log_metric("best_f1", best_f1)
        mlflow.log_param("best_threshold", best_threshold)

        print(f"{name} logged.")
        print(f"  ROC-AUC: {roc_auc:.6f}")
        print(f"  Best F1: {best_f1:.6f} @ threshold {best_threshold}")

# ---- Log each model ----
for model_name, file_name in oof_files.items():
    preds = np.load(os.path.join(NPY_DIR, file_name))
    log_metrics(model_name, preds)
