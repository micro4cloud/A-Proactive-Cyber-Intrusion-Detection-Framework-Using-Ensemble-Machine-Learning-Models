from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score
)
import json

def evaluate_model(model, X_test, y_test):
    y_pred = (model.predict(X_test) > 0.5).astype("int32")

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "mcc": float(matthews_corrcoef(y_test, y_pred)),
        "auc": float(roc_auc_score(y_test, y_pred))
    }

    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics