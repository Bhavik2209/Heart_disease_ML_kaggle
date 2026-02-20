import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from config import (
    EXPERIMENT_NAME,
    TARGET_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    BINARY_FEATURES,
    ORDINAL_FEATURES
)

from preprocessing import build_tree_preprocessor


# ---- Force same backend as your current setup ----
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(EXPERIMENT_NAME)


def train_random_forest():

    # -----------------------------
    # Load Data
    # -----------------------------
    df = pd.read_csv(r"D:\bacancy\Bacancy\Heart_disease\data\raw\train.csv")
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Absence": 0, "Presence": 1})

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # -----------------------------
    # Train / Validation Split
    # -----------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # -----------------------------
    # Build Pipeline
    # -----------------------------
    preprocessor = build_tree_preprocessor()

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", rf_model)
    ])

    # -----------------------------
    # Train
    # -----------------------------
    pipeline.fit(X_train, y_train)

    # -----------------------------
    # Predict
    # -----------------------------
    y_pred = pipeline.predict(X_val)
    y_prob = pipeline.predict_proba(X_val)[:, 1]

    # -----------------------------
    # Metrics
    # -----------------------------
    roc_auc = roc_auc_score(y_val, y_prob)
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    report = classification_report(y_val, y_pred)

    # -----------------------------
    # Log to MLflow
    # -----------------------------
    with mlflow.start_run(run_name="RandomForest_Baseline"):

        mlflow.log_param("model", "RandomForest")
        mlflow.log_param("n_estimators", 300)

        mlflow.log_metrics({
            "roc_auc": roc_auc,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

        # Save classification report as artifact
        os.makedirs("data/artifacts", exist_ok=True)
        report_path = "data/artifacts/random_forest_classification_report.txt"

        with open(report_path, "w") as f:
            f.write(report)

        mlflow.log_artifact(report_path)

        mlflow.sklearn.log_model(pipeline, "model")

    print("Random Forest training complete.")
    print(report)


if __name__ == "__main__":
    train_random_forest()
