import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from config import *
from models import get_lgb_model
from preprocessing import build_tree_preprocessor
from evaluate import compute_oof_auc

def train_lgb(X, y):

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="LightGBM_Final"):

        model = get_lgb_model()
        preprocessor = build_tree_preprocessor()

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        skf = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE
        )

        oof = np.zeros(len(X))

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

            pipeline.fit(X_tr, y_tr)
            oof[val_idx] = pipeline.predict_proba(X_val)[:, 1]

        auc = compute_oof_auc(y, oof)

        mlflow.log_metric("oof_roc_auc", auc)
        mlflow.sklearn.log_model(pipeline, "model")

        print("Final OOF AUC:", auc)
