import mlflow
from config import EXPERIMENT_NAME

mlflow.set_experiment(EXPERIMENT_NAME)

# ---------------------------
# LIGHTGBM MANUAL LOG
# ---------------------------
with mlflow.start_run(run_name="LightGBM_Manual"):

    mlflow.log_params({
        "model": "LightGBM",
        "n_estimators": 1600,
        "learning_rate": 0.02,
        "max_depth": 8,
        "num_leaves": 31,
        "min_child_samples": 40,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.5,
        "reg_lambda": 0.5
    })

    mlflow.log_metrics({
        "cv_accuracy_mean": 0.8887,
        "cv_precision_mean": 0.8820,
        "cv_recall_mean": 0.8678,
        "cv_f1_mean": 0.8748,
        "cv_roc_auc_mean": 0.9553,
        "cv_roc_auc_std": 0.0005,
        "oof_roc_auc": 0.955308731572325
    })

# ---------------------------
# XGBOOST MANUAL LOG
# ---------------------------
with mlflow.start_run(run_name="XGBoost_Manual"):
    mlflow.log_metric("oof_roc_auc", 0.9554845840220193)

# ---------------------------
# CATBOOST MANUAL LOG
# ---------------------------
with mlflow.start_run(run_name="CatBoost_Manual"):
    mlflow.log_metric("oof_roc_auc", 0.9554481745032924)

print("Manual runs logged successfully.")
