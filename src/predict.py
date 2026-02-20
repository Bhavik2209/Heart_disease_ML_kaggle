import pandas as pd
import mlflow.sklearn


def predict(test_path, model_uri):

    model = mlflow.sklearn.load_model(model_uri)

    test = pd.read_csv(test_path)
    test_ids = test["id"]
    X_test = test.drop(columns=["id"])

    probs = model.predict_proba(X_test)[:, 1]

    submission = pd.DataFrame({
        "id": test_ids,
        "Heart Disease": probs
    })

    submission.to_csv("submission.csv", index=False)
