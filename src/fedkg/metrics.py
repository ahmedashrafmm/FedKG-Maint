import numpy as np


def rmse(pred, true):
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def nasa_score(pred, true):
    """Asymmetric C-MAPSS score: late predictions are penalised more."""
    d = np.asarray(pred, float) - np.asarray(true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))
