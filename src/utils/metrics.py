"""평가 지표: MAE, RMSE, MAPE, AUROC, F1."""
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    mask = np.abs(y_true) > eps
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Symmetric MAPE — 분모가 작아도 안정적."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2 + eps
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE":   mae(y_true, y_pred),
        "RMSE":  rmse(y_true, y_pred),
        "MAPE":  mape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
    }


def classification_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    try:
        auroc = roc_auc_score(y_true.ravel(), y_prob.ravel())
    except ValueError:
        auroc = float("nan")
    f1 = f1_score(y_true.ravel(), y_pred.ravel(), zero_division=0)
    acc = float(np.mean(y_true.ravel() == y_pred.ravel()))
    return {"Accuracy": acc, "F1": f1, "AUROC": auroc}


def port_metrics(y_true: np.ndarray, y_pred: np.ndarray, port_names: list) -> dict:
    """항만별 RMSE 반환. y shape: (..., n_ports)."""
    results = {}
    for i, port in enumerate(port_names):
        results[port] = rmse(y_true[..., i], y_pred[..., i])
    return results
