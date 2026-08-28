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
