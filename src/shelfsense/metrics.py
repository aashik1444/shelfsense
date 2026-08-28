from __future__ import annotations

import numpy as np
import pandas as pd


def rmsse(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
    """Root Mean Squared Scaled Error for one series.

    The denominator is the in-sample one-step naive error on the TRAINING
    window only (never the full history) -- using the full history would
    let the denominator see data from inside the test fold, inflating the
    apparent improvement.
    """
    y_train = np.asarray(y_train, dtype=float)
    naive_diffs = np.diff(y_train)
    denom = np.mean(naive_diffs**2)
    if denom == 0:
        return np.nan
    num = np.mean((np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) ** 2)
    return float(np.sqrt(num / denom))


def weighted_rmsse(
    rmsse_per_series: pd.Series,
    weights: pd.Series,
) -> float:
    """Dollar-weighted mean of per-series RMSSE. `weights` is each series'
    dollar sales in the last 28 days of the training window -- the
    business-relevant aggregate, not a plain unweighted mean across series
    of wildly different revenue."""
    aligned = pd.DataFrame({"rmsse": rmsse_per_series, "weight": weights}).dropna()
    if aligned["weight"].sum() == 0:
        return np.nan
    return float(np.average(aligned["rmsse"], weights=aligned["weight"]))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean signed error: y_pred - y_true. Positive = over-forecasting on
    average, negative = under-forecasting. Reported because it matters for
    inventory and is rarely reported in forecasting writeups that only
    show MAE/RMSSE."""
    return float(np.mean(np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)))


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Pinball (quantile) loss at a given quantile alpha. Asymmetric: an
    under-forecast is penalized by alpha, an over-forecast by (1-alpha)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    diff = y_true - y_pred
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))
