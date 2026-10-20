"""C-MAPSS loading, per-factory (local) normalisation and windowing."""
import os
import numpy as np
import pandas as pd

COLS = ["unit", "cycle"] + [f"op{i}" for i in range(1, 4)] + [f"s{i}" for i in range(1, 22)]
# Fixed, shared feature set so every factory has the same input dimension.
SENSORS = [f"s{i}" for i in (2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21)]
FEATURES = ["op1", "op2", "op3"] + SENSORS


def _read(path):
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLS, engine="python")


def add_rul(df, cap):
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df["rul"] = (max_cycle - df["cycle"]).clip(upper=cap)
    return df


def _windows(df, window, last_only=False):
    X, y = [], []
    for _, g in df.groupby("unit"):
        feats = g[FEATURES].values.astype(np.float32)
        rul = g["rul"].values.astype(np.float32) if "rul" in g else None
        if len(feats) < window:  # left-pad short engines with their first row
            pad = np.repeat(feats[:1], window - len(feats), axis=0)
            feats = np.vstack([pad, feats])
            if rul is not None:
                rul = np.concatenate([np.repeat(rul[:1], window - len(rul)), rul])
        if last_only:
            X.append(feats[-window:])
            if rul is not None:
                y.append(rul[-1])
        else:
            for i in range(len(feats) - window + 1):
                X.append(feats[i:i + window])
                y.append(rul[i + window - 1])
    return np.stack(X), (np.array(y, dtype=np.float32) if y else None)


def load_factory(root, name, window=30, rul_cap=125):
    """Return train/test arrays for one factory. Normalisation stats are computed
    on this factory's training data only (raw data is never pooled)."""
    train = add_rul(_read(os.path.join(root, f"train_{name}.txt")), rul_cap)
    test = _read(os.path.join(root, f"test_{name}.txt"))
    true_rul = np.loadtxt(os.path.join(root, f"RUL_{name}.txt")).clip(max=rul_cap)

    mean = train[FEATURES].mean()
    std = train[FEATURES].std().replace(0, 1.0)
    train[FEATURES] = (train[FEATURES] - mean) / std
    test[FEATURES] = (test[FEATURES] - mean) / std

    Xtr, ytr = _windows(train, window)
    Xte, _ = _windows(test, window, last_only=True)
    return {"X_train": Xtr, "y_train": ytr, "X_test": Xte, "y_test": true_rul.astype(np.float32)}
