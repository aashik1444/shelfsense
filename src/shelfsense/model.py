from __future__ import annotations

from dataclasses import dataclass

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from shelfsense.config import Config

CATEGORICAL_FEATURES = ["item_id", "dept_id", "store_id", "state_id", "event_type_1"]

NUMERIC_FEATURES = [
    "days_since_first_sale",
    "lag_28", "lag_35", "lag_42", "lag_56", "lag_364",
    "roll_mean_7", "roll_mean_28", "roll_mean_56", "roll_mean_180",
    "roll_std_28", "roll_std_56", "roll_max_28", "roll_zero_share_28",
    "zero_run_length",
    "sell_price", "price_vs_item_median_8w", "price_vs_dept_median_week",
    "price_changed_flag", "price_momentum_4w",
    "dow", "day_of_month", "week_of_year", "month", "year", "is_weekend",
    "snap", "days_to_next_event", "days_since_last_event", "is_christmas",
    "days_ahead",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_training_matrix(
    con: duckdb.DuckDBPyConnection,
    config: Config,
    origin_date: pd.Timestamp,
) -> pd.DataFrame:
    """Direct multi-horizon training matrix for one fold, built entirely in
    DuckDB (the explode-and-join over ~700 anchor days x 4,065 series x 28
    horizons is far too large to do in pandas).

    Each anchor row in feature_matrix at date t carries features already
    shifted to be known at t-28 (the h=28 leakage shift baked into the SQL
    layer). We reuse that same anchor for every horizon: for k = 1..28,
    the training example predicts units at (t - 28 + k) using the SAME
    features (known at t-28), with days_ahead=k marking which offset from
    the anchor is being predicted. This is what makes it "direct" rather
    than "recursive" -- one model, days_ahead as a feature, no chaining of
    predictions into later predictions.

    Anchors are restricted to the fold's training window so this function
    can be reused unchanged for both fold-training and single-shot scoring.

    Anchor dates are subsampled to every `anchor_stride_days`-th calendar
    day (anchored to the fold's own origin, so the origin date itself is
    always included) rather than using every day in the 730-day window.
    Direct multi-horizon already explodes each anchor into 28 rows (one
    per days_ahead); using every day on top of that produces ~80M rows for
    a single fold, which is neither necessary (adjacent daily anchors are
    highly redundant -- their lag/rolling features barely differ) nor
    practical to train repeatedly across 3 folds x 8 hyperparameter
    configs. Striding by 7 days still covers the full 2-year window and
    every season/weekday, at a fraction of the row count.
    """
    train_start = origin_date - pd.Timedelta(days=config.train_days - 1)

    query = """
        WITH anchors AS (
            SELECT * FROM feature_matrix
            WHERE date BETWEEN $train_start AND $origin_date
              AND roll_mean_28 IS NOT NULL
              AND DATE_DIFF('day', date, $origin_date::DATE) % $stride = 0
        ),
        horizons AS (
            SELECT UNNEST(generate_series(1, $horizon)) AS days_ahead
        )
        SELECT
            a.* EXCLUDE (date, units),
            h.days_ahead,
            a.date + CAST(h.days_ahead - $horizon AS INTEGER) AS target_date,
            t.units AS target_units
        FROM anchors a
        CROSS JOIN horizons h
        JOIN feature_matrix t
            ON t.id = a.id
           AND t.date = a.date + CAST(h.days_ahead - $horizon AS INTEGER)
    """
    return con.execute(
        query,
        {
            "train_start": train_start.date(),
            "origin_date": origin_date.date(),
            "horizon": config.horizon,
            "stride": config.anchor_stride_days,
        },
    ).df()


@dataclass
class TrainedModel:
    booster: lgb.Booster
    objective: str
    features: list[str]
    categorical_features: list[str]


def train(
    train_df: pd.DataFrame,
    objective: str,
    seed: int,
    num_leaves: int = 128,
    learning_rate: float = 0.1,
    min_data_in_leaf: int = 100,
    quantile_alpha: float | None = None,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 30,
    valid_df: pd.DataFrame | None = None,
) -> TrainedModel:
    for col in CATEGORICAL_FEATURES:
        train_df[col] = train_df[col].astype("category")
        if valid_df is not None:
            valid_df[col] = pd.Categorical(valid_df[col], categories=train_df[col].cat.categories)

    params = {
        "objective": objective,
        "num_leaves": num_leaves,
        "learning_rate": learning_rate,
        "min_data_in_leaf": min_data_in_leaf,
        "seed": seed,
        "deterministic": True,
        "verbose": -1,
    }
    if objective == "tweedie":
        params["tweedie_variance_power"] = 1.1
    if objective == "quantile":
        if quantile_alpha is None:
            raise ValueError("quantile_alpha required for objective='quantile'")
        params["alpha"] = quantile_alpha

    # free_raw_data=True (the default): once LightGBM has built its internal
    # binary representation it doesn't need the pandas copy anymore, and
    # keeping it (free_raw_data=False) roughly doubles peak memory per
    # config -- material when this runs 8 times in one process for the
    # tuning grid.
    dtrain = lgb.Dataset(
        train_df[FEATURES], label=train_df["target_units"],
        categorical_feature=CATEGORICAL_FEATURES,
    )
    callbacks = []
    valid_sets = [dtrain]
    valid_names = ["train"]
    if valid_df is not None:
        dvalid = lgb.Dataset(
            valid_df[FEATURES], label=valid_df["target_units"],
            categorical_feature=CATEGORICAL_FEATURES, reference=dtrain,
        )
        valid_sets.append(dvalid)
        valid_names.append("valid")
        callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))

    booster = lgb.train(
        params, dtrain,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets, valid_names=valid_names,
        callbacks=callbacks,
    )
    return TrainedModel(booster=booster, objective=objective, features=FEATURES, categorical_features=CATEGORICAL_FEATURES)


def predict(model: TrainedModel, df: pd.DataFrame) -> np.ndarray:
    df = df.copy()
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("category")
    preds = model.booster.predict(df[model.features])
    return np.maximum(preds, 0.0)
