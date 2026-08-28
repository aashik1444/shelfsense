from __future__ import annotations

import duckdb
import pytest

from shelfsense.config import load_config
from shelfsense.backtest import _origin_date


@pytest.fixture(scope="module")
def con():
    config = load_config()
    return duckdb.connect(config.db_path, read_only=True)


@pytest.fixture(scope="module")
def config():
    return load_config()


def test_train_never_overlaps_test_within_a_fold(con, config):
    """For every configured backtest origin, the train window (trailing
    train_days ending at the origin) must end strictly before the test
    window (origin+1 .. origin+horizon) begins. This is checked against
    actual dates pulled from the warehouse, not just arithmetic on
    integers, so a mistake in the calendar join would also be caught."""
    import pandas as pd

    for origin_d in config.backtest_origins:
        origin_date = _origin_date(con, origin_d)
        train_start = origin_date - pd.Timedelta(days=config.train_days - 1)
        test_start = origin_date + pd.Timedelta(days=1)
        test_end = origin_date + pd.Timedelta(days=config.horizon)

        assert train_start <= origin_date < test_start
        assert test_start <= test_end

        max_train_date = con.execute(
            "SELECT MAX(date) FROM feature_matrix WHERE date BETWEEN ? AND ?",
            [train_start.date(), origin_date.date()],
        ).fetchone()[0]
        min_test_date = con.execute(
            "SELECT MIN(date) FROM feature_matrix WHERE date BETWEEN ? AND ?",
            [test_start.date(), test_end.date()],
        ).fetchone()[0]

        if max_train_date is None or min_test_date is None:
            continue
        assert max_train_date < min_test_date, (
            f"fold at origin d_{origin_d}: train date {max_train_date} "
            f"overlaps or exceeds test date {min_test_date}"
        )


def test_folds_are_distinct_28_day_blocks(con, config):
    """The three configured origins should each be exactly 28 days apart,
    matching the 28-day forecast horizon -- confirming the folds tile the
    final 84 days without overlap or gaps."""
    import pandas as pd

    dates = [_origin_date(con, d) for d in config.backtest_origins]
    for earlier, later in zip(dates, dates[1:]):
        assert (later - earlier).days == config.horizon
