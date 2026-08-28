from __future__ import annotations

import argparse

import duckdb
import pandas as pd

from shelfsense.baselines import BASELINES
from shelfsense.config import Config, load_config
from shelfsense.metrics import bias, mae, rmsse, weighted_rmsse


def _origin_date(con: duckdb.DuckDBPyConnection, origin_d: int) -> pd.Timestamp:
    result = con.execute(
        "SELECT date FROM raw_calendar WHERE d = ?", [f"d_{origin_d}"]
    ).fetchone()
    return pd.Timestamp(result[0])


def run_fold(
    con: duckdb.DuckDBPyConnection,
    config: Config,
    origin_d: int,
    model_name: str,
) -> dict:
    origin_date = _origin_date(con, origin_d)
    train_start = origin_date - pd.Timedelta(days=config.train_days - 1)
    test_start = origin_date + pd.Timedelta(days=1)
    test_end = origin_date + pd.Timedelta(days=config.horizon)
    weight_start = origin_date - pd.Timedelta(days=config.horizon - 1)

    train_df = con.execute(
        "SELECT id, date, units FROM feature_matrix WHERE date BETWEEN ? AND ?",
        [train_start.date(), origin_date.date()],
    ).df()

    test_df = con.execute(
        """
        SELECT id, date, units, lag_28, lag_35, roll_mean_28, roll_zero_share_28
        FROM feature_matrix
        WHERE date BETWEEN ? AND ?
        """,
        [test_start.date(), test_end.date()],
    ).df()

    weight_df = con.execute(
        """
        SELECT id, SUM(units * sell_price) AS dollar_sales
        FROM feature_matrix
        WHERE date BETWEEN ? AND ?
        GROUP BY id
        """,
        [weight_start.date(), origin_date.date()],
    ).df()
    weights = weight_df.set_index("id")["dollar_sales"]

    predict_fn = BASELINES[model_name]
    test_df = test_df.copy()
    test_df["y_pred"] = predict_fn(test_df)

    train_by_id = {
        series_id: group["units"].to_numpy()
        for series_id, group in train_df.groupby("id", sort=False)
    }

    per_series_rmsse = {}
    for series_id, group in test_df.groupby("id", sort=False):
        train_series = train_by_id.get(series_id)
        if train_series is None or len(train_series) < 2:
            continue
        group_valid = group.dropna(subset=["y_pred"])
        if group_valid.empty:
            continue
        per_series_rmsse[series_id] = rmsse(
            group_valid["units"].to_numpy(),
            group_valid["y_pred"].to_numpy(),
            train_series,
        )

    rmsse_series = pd.Series(per_series_rmsse)
    valid_test = test_df.dropna(subset=["y_pred"])

    return {
        "origin_d": origin_d,
        "origin_date": str(origin_date.date()),
        "weighted_rmsse": weighted_rmsse(rmsse_series, weights),
        "mae": mae(valid_test["units"], valid_test["y_pred"]),
        "bias": bias(valid_test["units"], valid_test["y_pred"]),
        "n_series": len(per_series_rmsse),
    }


def run_backtest(model_name: str, config: Config | None = None) -> pd.DataFrame:
    config = config or load_config()
    con = duckdb.connect(config.db_path, read_only=True)
    rows = [run_fold(con, config, origin, model_name) for origin in config.backtest_origins]
    con.close()
    df = pd.DataFrame(rows)
    df["model"] = model_name
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(BASELINES.keys()))
    args = parser.parse_args()

    results = run_backtest(args.model)
    print(results.to_string(index=False))
    print()
    print("Mean across folds:")
    print(results[["weighted_rmsse", "mae", "bias"]].mean())


if __name__ == "__main__":
    main()
