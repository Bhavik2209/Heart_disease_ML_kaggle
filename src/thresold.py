import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def threshold_analysis(y_true, y_prob):
    thresholds = np.arange(0.1, 0.95, 0.05)

    results = []

    for t in thresholds:
        preds = (y_prob >= t).astype(int)

        results.append({
            "threshold": float(t),
            "precision": precision_score(y_true, preds),
            "recall": recall_score(y_true, preds),
            "f1": f1_score(y_true, preds)
        })

    return results
