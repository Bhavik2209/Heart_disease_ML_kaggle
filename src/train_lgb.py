import mlflow
import mlflow.sklearn
from config import EXPERIMENT_NAME
from models import get_lgb_model
from cross_validation import run_oof

def train_lgb(X, y):

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="LightGBM_Optuna_Final"):

        model = get_lgb_model()

        oof, auc = run_oof(model, X, y)

        mlflow.log_param("model", "LightGBM")
        mlflow.log_metric("oof_roc_auc", auc)

        mlflow.sklearn.log_model(model, "model")

        print("OOF ROC-AUC:", auc)
