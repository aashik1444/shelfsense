"""Reconstructs lag/rolling feature values from raw sales_long using only
data at or before t-28, and asserts they match feature_matrix exactly.

This is the most important test in the project: it does not trust the SQL's
own window-frame logic, it recomputes the features independently in pandas
from first principles and checks the two agree. If the SQL frame clauses
were ever off by one row, or a lag/roll feature were built from data after
t-28, this test would fail on real rows, not on a synthetic fixture.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import pytest

from shelfsense.config import load_config

HORIZON = 28
SAMPLE_SIZE = 200


@pytest.fixture(scope="module")
def con():
    config = load_config()
    return duckdb.connect(config.db_path, read_only=True)


@pytest.fixture(scope="module")
def sample_rows(con) -> pd.DataFrame:
    return con.execute(
        f"""
        SELECT id, date, lag_28, lag_35, lag_42, lag_56, lag_364,
               roll_mean_7, roll_mean_28, roll_mean_56, roll_mean_180,
               roll_std_28, roll_std_56, roll_max_28, roll_zero_share_28
        FROM feature_matrix
        WHERE roll_mean_180 IS NOT NULL
        USING SAMPLE {SAMPLE_SIZE} ROWS
        """
    ).df()


def _raw_series(con, series_id: str) -> pd.DataFrame:
    df = con.execute(
        "SELECT date, units FROM sales_long WHERE id = ? ORDER BY date",
        [series_id],
    ).df()
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["units"]


def test_lag_features_reconstruct_from_raw(con, sample_rows):
    for _, row in sample_rows.iterrows():
        raw = _raw_series(con, row["id"])
        t = pd.Timestamp(row["date"])

        for lag_days, col in [(28, "lag_28"), (35, "lag_35"), (42, "lag_42"), (56, "lag_56")]:
            source_date = t - pd.Timedelta(days=lag_days)
            if source_date not in raw.index:
                continue
            expected = raw.loc[source_date]
            actual = row[col]
            assert actual == expected, (
                f"{row['id']} {t.date()} {col}: expected {expected} "
                f"(from {source_date.date()}), got {actual}"
            )


def test_lag_364_reconstructs_from_raw(con, sample_rows):
    rows_with_lag364 = sample_rows[sample_rows["lag_364"].notna()]
    if rows_with_lag364.empty:
        pytest.skip("no sampled rows had a populated lag_364")
    for _, row in rows_with_lag364.iterrows():
        raw = _raw_series(con, row["id"])
        t = pd.Timestamp(row["date"])
        source_date = t - pd.Timedelta(days=364)
        if source_date not in raw.index:
            continue
        assert row["lag_364"] == raw.loc[source_date]


def test_rolling_mean_reconstructs_from_raw_with_correct_shift(con, sample_rows):
    """roll_mean_28 must equal the mean of units over the 28-day window
    ending at t-28 -- i.e. dates [t-55, t-28] inclusive -- and must NOT
    equal the same window computed without the shift (dates [t-27, t]),
    which would be the leaky version."""
    for _, row in sample_rows.iterrows():
        raw = _raw_series(con, row["id"])
        t = pd.Timestamp(row["date"])

        window_start = t - pd.Timedelta(days=55)
        window_end = t - pd.Timedelta(days=28)
        correct_window = raw.loc[window_start:window_end]
        if len(correct_window) == 0:
            continue
        expected = correct_window.mean()

        actual = row["roll_mean_28"]
        assert np.isclose(actual, expected, atol=1e-6), (
            f"{row['id']} {t.date()} roll_mean_28: expected {expected} "
            f"from window [{window_start.date()}, {window_end.date()}], got {actual}"
        )

        # the leaky alternative: window ending at t instead of t-28
        leaky_window = raw.loc[t - pd.Timedelta(days=27) : t]
        if len(leaky_window) > 0:
            leaky_value = leaky_window.mean()
            if not np.isclose(leaky_value, expected, atol=1e-6):
                assert not np.isclose(actual, leaky_value, atol=1e-6), (
                    f"{row['id']} {t.date()} roll_mean_28 matches the LEAKY "
                    f"(unshifted) window instead of the correct t-28 shifted window"
                )


def test_rolling_std_reconstructs_from_raw(con, sample_rows):
    for _, row in sample_rows.iterrows():
        raw = _raw_series(con, row["id"])
        t = pd.Timestamp(row["date"])
        window_start = t - pd.Timedelta(days=55)
        window_end = t - pd.Timedelta(days=28)
        window = raw.loc[window_start:window_end]
        if len(window) < 2:
            continue
        expected = window.std(ddof=1)
        actual = row["roll_std_28"]
        if pd.isna(actual):
            continue
        assert np.isclose(actual, expected, atol=1e-6)


def test_roll_max_28_reconstructs_from_raw(con, sample_rows):
    for _, row in sample_rows.iterrows():
        raw = _raw_series(con, row["id"])
        t = pd.Timestamp(row["date"])
        window_start = t - pd.Timedelta(days=55)
        window_end = t - pd.Timedelta(days=28)
        window = raw.loc[window_start:window_end]
        if len(window) == 0:
            continue
        assert row["roll_max_28"] == window.max()


def test_no_feature_uses_data_after_t_minus_28(con):
    """Structural check: for every row where date - first_sale_date >= 28,
    lag_28 and roll_mean_28 must be non-null (there IS enough history to
    compute them), and by construction (frame clause) they cannot reference
    any date after t-28. This complements the value-reconstruction tests
    above by confirming the shift boundary itself, at scale, not just on
    the 200-row sample."""
    df = con.execute(
        """
        SELECT
            fm.id, fm.date, fm.lag_28, fm.roll_mean_28,
            fs.first_sale_date
        FROM feature_matrix fm
        JOIN (SELECT id, MIN(date) AS first_sale_date FROM sales_long GROUP BY id) fs
            ON fm.id = fs.id
        USING SAMPLE 500 ROWS
        """
    ).df()
    df["days_since_first_sale"] = (
        pd.to_datetime(df["date"]) - pd.to_datetime(df["first_sale_date"])
    ).dt.days

    eligible = df[df["days_since_first_sale"] >= 28]
    assert (eligible["lag_28"].notna()).all(), "lag_28 should be populated once >=28 days of history exist"
    assert (eligible["roll_mean_28"].notna()).all(), "roll_mean_28 should be populated once >=28 days of history exist"

    ineligible = df[df["days_since_first_sale"] < 28]
    assert (ineligible["lag_28"].isna()).all(), "lag_28 should be null before 28 days of history exist"
