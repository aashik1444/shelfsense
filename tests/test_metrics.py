from __future__ import annotations

import numpy as np

from shelfsense.metrics import bias, mae, pinball_loss, rmsse, weighted_rmsse
import pandas as pd


def test_rmsse_perfect_forecast_is_zero():
    y_train = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4], dtype=float)
    y_true = np.array([3, 4, 5], dtype=float)
    y_pred = y_true.copy()
    assert rmsse(y_true, y_pred, y_train) == 0.0


def test_rmsse_of_naive_forecast_on_training_tail_is_near_one():
    rng = np.random.default_rng(42)
    y_train = rng.poisson(lam=5, size=200).astype(float)

    # naive forecast on the training tail: predict y[t] with y[t-1]
    y_true = y_train[1:]
    y_pred = y_train[:-1]

    score = rmsse(y_true, y_pred, y_train)
    assert 0.9 < score < 1.1


def test_rmsse_zero_variance_train_returns_nan():
    y_train = np.array([5.0, 5.0, 5.0, 5.0])
    y_true = np.array([5.0])
    y_pred = np.array([6.0])
    assert np.isnan(rmsse(y_true, y_pred, y_train))


def test_weighted_rmsse_matches_manual_weighted_average():
    rmsse_per_series = pd.Series({"a": 1.0, "b": 2.0, "c": 3.0})
    weights = pd.Series({"a": 10.0, "b": 0.0, "c": 10.0})
    result = weighted_rmsse(rmsse_per_series, weights)
    expected = (1.0 * 10.0 + 3.0 * 10.0) / 20.0
    assert np.isclose(result, expected)


def test_mae_basic():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 4, 2])
    assert mae(y_true, y_pred) == np.mean([0, 2, 1])


def test_bias_positive_means_overforecast():
    y_true = np.array([1, 1, 1])
    y_pred = np.array([2, 2, 2])
    assert bias(y_true, y_pred) == 1.0


def test_bias_negative_means_underforecast():
    y_true = np.array([2, 2, 2])
    y_pred = np.array([1, 1, 1])
    assert bias(y_true, y_pred) == -1.0


def test_pinball_loss_zero_for_perfect_forecast():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = y_true.copy()
    assert pinball_loss(y_true, y_pred, 0.5) == 0.0


def test_pinball_loss_asymmetric_at_high_quantile():
    """At alpha=0.9, under-forecasting should be penalized more than
    over-forecasting by the same absolute amount."""
    y_true = np.array([10.0])
    under_forecast_loss = pinball_loss(y_true, np.array([8.0]), 0.9)
    over_forecast_loss = pinball_loss(y_true, np.array([12.0]), 0.9)
    assert under_forecast_loss > over_forecast_loss
