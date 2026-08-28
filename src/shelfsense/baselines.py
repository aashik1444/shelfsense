from __future__ import annotations

import pandas as pd


def naive(df: pd.DataFrame) -> pd.Series:
    """Last observed value. Horizon-legal because it's exactly lag_28 --
    the most recent value known at forecast time for a 28-day-ahead
    forecast is the value 28 days back, not "yesterday"."""
    return df["lag_28"]


def seasonal_naive_7(df: pd.DataFrame) -> pd.Series:
    """Same weekday last week, but taken from lag_35 (28 + 7), not lag_7.
    lag_7 would use data from inside the forecast horizon and is not
    horizon-legal for a 28-day-ahead forecast; lag_35 is the freshest
    "one week before the most recent horizon-legal day" value available."""
    return df["lag_35"]


def mean_28(df: pd.DataFrame) -> pd.Series:
    """Trailing 28-day mean, already computed with the correct t-28 shift
    in the SQL feature layer."""
    return df["roll_mean_28"]


def croston_lite(df: pd.DataFrame) -> pd.Series:
    """Croston-style intermittent-demand forecast: the average SIZE of a
    non-zero demand event, over the horizon-legal trailing 28-day window.
    roll_mean_28 already equals mean_nonzero_size * nonzero_rate (zeros
    contribute nothing to the sum), so dividing it back out by the
    nonzero rate recovers the average non-zero demand size. This is a
    simplified single-pass approximation of Croston's method (no separate
    inter-demand-interval smoothing), appropriate for a baseline rather
    than a production intermittent-demand model."""
    nonzero_rate = 1 - df["roll_zero_share_28"]
    return df["roll_mean_28"] / nonzero_rate.replace(0, pd.NA)


BASELINES = {
    "naive": naive,
    "seasonal_naive_7": seasonal_naive_7,
    "mean_28": mean_28,
    "croston_lite": croston_lite,
}
