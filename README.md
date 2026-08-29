# shelfsense

A 28-day-ahead demand forecasting pipeline for retail store-item series, built entirely on DuckDB SQL feature engineering and a global LightGBM model, with a leakage-safety test suite that reconstructs every engineered feature from raw data.

## The business problem

A category manager needs to know how many units of each item to have on the shelf 28 days out. Two things make this hard in practice, not just in theory:

- **The horizon is 28 days, not 1.** By the time an order is placed, the freshest sales data available is already a month stale relative to the day the units need to be on the shelf. Any feature built from "yesterday's sales" is not usable — every signal has to be computed as of `t-28`.
- **Demand is intermittent.** Most store-item-days sell zero or a handful of units. A model (or a naive baseline) that ignores this and just predicts a mean will be systematically wrong in a specific direction depending on how it handles zeros — this project measures that directly rather than assuming it away.

This repo builds the forecasting half of that problem: a leakage-safe feature pipeline and a validated model, backtested honestly against baselines that are hard to beat.

## Dataset and scope

Walmart M5 (`sales_train_evaluation.csv`, `calendar.csv`, `sell_prices.csv`). Scoped to:

```
stores : CA_3, TX_1, WI_2       (one per state)
depts  : FOODS_3, HOUSEHOLD_1   (two highest-volume departments)
```

≈4,065 series (1,355 items × 3 stores), 2011-01-29 through 2016-05-22.

**Why not all 30,490 series?** The SQL feature engineering, the leakage-safety design, and the LightGBM architecture don't change with series count — `stores`/`depts` are config values, not code. What changes is backtest iteration speed. The saved compute went into a 3-fold rolling-origin backtest, an 8-config hyperparameter grid with full grid logging, and honest reporting against the strongest baseline rather than the weakest one. The store list can be widened in `config.yaml` with no code change.

## Architecture

```
data/raw/*.csv (M5 CSVs)
        │
        ▼
sql/01_load_raw.sql          raw_calendar, raw_sales, raw_prices
        │
        ▼
sql/02_sales_long.sql        UNPIVOT wide→long, scope filter,
                              calendar + price join, first-sale trim
        │
        ▼
        sales_long            (6,166,395 rows, one row per store-item-day)
        │
        ▼
sql/03_calendar_features.sql  known-at-t calendar features
sql/04_price_features.sql     known-at-t price features
sql/05_feature_matrix.sql     lag/rolling features, shifted via window
                               FRAME clauses (the leakage-critical file)
        │
        ▼
        feature_matrix        (same row count as sales_long — no fanout)
        │
        ├──► src/shelfsense/backtest.py     baselines: naive, seasonal_naive_7,
        │                                    mean_28, croston_lite
        │
        └──► src/shelfsense/model.py         direct multi-horizon training matrix
             src/shelfsense/lgbm_backtest.py  LightGBM (Tweedie), 8-config grid,
                                               3-fold rolling-origin backtest
                     │
                     ▼
             models/lgbm_tweedie_final.txt   (gitignored build artifact)
```

## Key engineering decisions

**The leakage shift is a window FRAME clause, not a lag-then-roll.** Every lag and rolling feature in `05_feature_matrix.sql` is built with `ROWS BETWEEN 55 PRECEDING AND 28 PRECEDING` (or equivalent bounds) — the frame's upper bound *is* the 28-day shift, structurally, not a `.shift(28)` a later edit could accidentally drop. `tests/test_leakage.py` independently reconstructs these values from raw data in pandas and confirms both that the correct value matches and that it does *not* match what the leaky, unshifted window would have produced.

**Direct multi-horizon, not recursive.** One model predicts `units` at any offset within the 28-day window, with `days_ahead` as a feature, trained by exploding each anchor row's already-shifted features into 28 training examples (one per horizon day). Recursive forecasting — feeding day-1's prediction back in to predict day-2 — compounds error across 28 steps; direct multi-horizon avoids that by conditioning every prediction on real observed history.

**One global model, not one per series.** 4,065 individual models can't share cross-series structure (a SNAP effect, an event effect) and handle short-history items poorly. A single model learns those effects once from all series' evidence.

**Tweedie objective.** Retail unit demand is non-negative, zero-inflated, and right-skewed — Tweedie (`variance_power=1.1`) is the standard distributional fit for that shape. On this particular slice, plain L2 scored marginally better (0.8275 vs 0.8279, a 0.05% gap) — reported honestly rather than switched to, since the gap is within noise and the distributional argument for Tweedie doesn't depend on winning by a specific margin on one slice.

**Rolling-origin backtest, 3 folds.** Each fold trains on a trailing 730-day window and tests on the 28 days immediately after a fixed origin (`d_1857`, `d_1885`, `d_1913`), so performance is checked across three different periods rather than one arbitrary split. The RMSSE scaling denominator is computed from each fold's training window only, never the full series history.

## Results

3-fold mean, dollar-weighted RMSSE:

| model | weighted RMSSE | mean MAE | mean bias |
|---|---|---|---|
| naive | 1.066 | 1.939 | +0.012 |
| seasonal_naive_7 | 1.084 | 1.976 | -0.005 |
| mean_28 | 0.829 | 1.629 | -0.035 |
| croston_lite | 0.895 | 2.019 | +0.857 |
| **LightGBM (Tweedie, tuned)** | **0.785** | **1.557** | **+0.002** |

**LightGBM beats `mean_28` — the strongest baseline — by 5.2%.** Against the weaker `seasonal_naive_7` baseline the improvement is 27.6%, but `mean_28` is the honest bar: it already captures most of the predictable signal in this stable, slow-moving product mix. The model's real edge is using price, calendar, SNAP, and event information that `mean_28` structurally cannot use at all, not outlearning the trend/level itself. The 5.2% number, not the 27.6% one, is what should be quoted first.

Full per-fold tables (hyperparameter grid, objective comparison, per-fold backtest) are in `reports/results.md`.

## Validation

- **`tests/test_leakage.py`** (6 tests) — the most important test file in the repo. It reconstructs `lag_28`, `lag_35`, `lag_42`, `lag_56`, `lag_364`, `roll_mean_28`, `roll_std_28`, and `roll_max_28` independently in pandas from `sales_long`, for a 200-row sample, and asserts exact/near-exact agreement with `feature_matrix`. One test explicitly checks the reconstructed value does *not* match what the leaky (unshifted) window would have produced — a positive match alone can pass by coincidence on slow-moving series; the negative check is what actually proves the shift is correctly implemented.
- **`tests/test_metrics.py`** (9 tests) — validates the metrics functions in isolation: RMSSE of a perfect forecast is 0, RMSSE of a naive forecast on its own training tail is ≈1, weighted RMSSE matches a manual weighted average, bias sign convention, pinball loss asymmetry.
- **`tests/test_splits.py`** (2 tests) — confirms, against real calendar-joined dates pulled from the warehouse (not index arithmetic), that no train date in a fold is ever ≥ any test date in the same fold, and that the three configured origins are exactly 28 days apart.

**17 tests total, all passing** as of this scope freeze.

## How to run it

Requires Python ≥3.11 and the M5 CSVs (`calendar.csv`, `sales_train_evaluation.csv`, `sell_prices.csv`, `sample_submission.csv`) placed in `data/raw/`.

```bash
python -m venv .venv
.venv/Scripts/pip install -e .          # or .venv/bin/pip on macOS/Linux

python -m shelfsense.warehouse build     # ~2-3 min: builds sales_long + feature_matrix
python -m shelfsense.backtest --model mean_28   # ~15s: one baseline, all 3 folds
pytest tests/                            # ~8s: 17 tests
```

The LightGBM tuning grid, objective comparison, and 3-fold backtest that produced the numbers above are in `src/shelfsense/lgbm_backtest.py` (`tune_on_fold1`, `run_lgbm_backtest`, `save_final_model`) — each full 3-fold LightGBM run takes ~12-15 minutes on a 16GB machine with `anchor_stride_days=14` (see `docs/decisions.md` Entry 5 for why that parameter exists).

## What I'd do next

The following are explicitly **not implemented** in this repo. They were scoped out when the project was frozen at M4, not attempted and abandoned:

- **Promotional uplift measurement (DiD, event study, power analysis).** Whether a price promotion in this data actually caused a sales lift, versus the item would have sold anyway, is unanswered — no causal design has been built.
- **SHAP attribution and error segmentation.** The model has not been explained feature-by-feature, and there's no breakdown of where it over- or under-performs (by department, volume decile, intermittency, SNAP day, event week).
- **Cost-of-error / newsvendor framing.** The 5.2%/27.6% RMSSE improvements are not yet converted into a dollar figure. `src/shelfsense/cost.py` and `src/shelfsense/plots.py` exist as unintegrated scaffolding from work that was stopped mid-milestone — they are not wired into any pipeline output and should not be treated as validated.
- **The fold-to-fold bias sign flip is unresolved.** Mean bias across the 3 folds is +0.002 (near-neutral), but that average hides a real pattern: the model under-forecasts on fold 1 (bias -0.025) and over-forecasts on folds 2-3 (+0.007, +0.023). No investigation has been done into why the sign flips or whether it's tied to a specific segment, season, or the `anchor_stride_days=14` subsampling.
