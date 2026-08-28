# Decisions log

## 2026-08-28 — Entry 1: Scope lock and data verification

Scoped to 3 stores (CA_3, TX_1, WI_2) x 2 depts (FOODS_3, HOUSEHOLD_1), ~4,000
series, out of the full 30,490-series M5 dataset. Reason: the modelling and
SQL-engineering decisions are identical at 4k and 30k series; spending the
saved compute on validation rigour (3-fold rolling-origin backtest, leakage
tests, cost sensitivity) is higher-value for an interview than raw row count.
The pipeline is config-driven (config.yaml), so widening scope is a one-line
change if needed later.

Data verification (sanity_check.py, run against data/raw/):

| file | rows | cols | notes |
|---|---|---|---|
| calendar.csv | 1,969 | 14 | 2011-01-29 -> 2016-06-19, 7,542 nulls (expected: event_name/type columns are mostly null, only ~2% of days have events) |
| sales_train_evaluation.csv | 30,490 | 1,947 | d_1 ... d_1941, zero nulls |
| sell_prices.csv | 6,841,121 | 4 | zero nulls in the raw table; price is null-by-omission (row simply absent) before an item is stocked, not a null cell |
| sample_submission.csv | 60,980 | 29 | not used by this project |

All counts match the expected values in MILESTONES.md M0 exactly.

`sales_train_validation.csv` (stops at d_1913) was deleted from data/raw/ per
spec — this project uses only `sales_train_evaluation.csv` (through d_1941),
and keeping both invites loading the wrong one by mistake.

## 2026-08-28 — Entry 2: DuckDB warehouse (M1)

Built `sales_long` via DuckDB `UNPIVOT ... ON COLUMNS(* EXCLUDE (...))`
rather than a generated UNION ALL, scoped inline to the configured
stores/depts, LEFT JOINed to prices, with pre-launch zero-runs trimmed via
an explicit `first_sale_date` per series (not via join semantics). Full
rebuild: 95s, 4,065 distinct series, zero null prices post-trim.

Spec note: MILESTONES.md's M1 DoD says the date range should end
2016-06-19; `sales_long` correctly ends 2016-05-22 (the last date with
actual sales in `sales_train_evaluation.csv` / `d_1941`). The extra 28 days
in `calendar.csv` (through `d_1969`) are the forecast horizon itself —
calendar/price/SNAP features to predict against, with no sales ground
truth — not additional training history. Did not force a match; the spec
text conflated the calendar table's range with the sales table's range.
See `docs/build_journal.md` M1 section for the full reasoning.

## 2026-08-28 — Entry 3: SQL feature layer (M2)

Built `feature_matrix` (6,166,395 rows, same row count as `sales_long` — no
fanout) via three SQL files: `03_calendar_features.sql`,
`04_price_features.sql`, `05_feature_matrix.sql`. All lag/rolling features
shifted by the window FRAME clause (`ROWS BETWEEN 55 PRECEDING AND 28
PRECEDING` etc.), not by lag-then-roll. `tests/test_leakage.py` reconstructs
every lag and rolling feature independently in pandas from `sales_long` for
a 200-row sample and asserts exact/near-exact match, including one test that
explicitly checks the value does NOT match the leaky (unshifted) window.
All 6 tests pass. Full `feature_matrix` build: 35s (from `sales_long`),
well under the 90s bar; full warehouse rebuild end-to-end: ~138s.

Known-at-forecast-time table:

| feature | known at t? | reason |
|---|---|---|
| lag_28, lag_35, lag_42, lag_56, lag_364 | yes (by construction) | shifted >=28 days via LAG(col, n) |
| roll_mean_7/28/56/180, roll_std_28/56, roll_max_28, roll_zero_share_28 | yes (by construction) | window frame ends at t-28 |
| zero_run_length | yes (by construction) | run length as of t-28, computed then shifted |
| sell_price, price_vs_item_median_8w, price_vs_dept_median_week, price_changed_flag, price_momentum_4w | yes | price is a retailer plan, set in advance |
| dow, day_of_month, week_of_year, month, year, is_weekend | yes | calendar facts about date t itself |
| snap | yes | SNAP eligibility is a known monthly calendar rule |
| event_type_1, days_to_next_event, days_since_last_event | yes | published holiday/event calendar |
| is_christmas | yes | calendar fact |
| item_id, dept_id, cat_id, store_id, state_id | yes | series identity, static |
| days_since_first_sale | yes | fact about the series' own age |

Null rates all land where expected: lag_28/roll_* null only for the first
28 days of each series (113,820 rows = 28 x 4,065 series, exact); lag_364
null for series younger than a year (23.95%); event_type_1 null on the
~92% of days with no event (correct — most days have no holiday/event).
No feature was imputed with zero; LightGBM handles nulls natively (M4).
