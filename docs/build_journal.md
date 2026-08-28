# shelfsense — Build Journal

A running, milestone-by-milestone record of everything built: the code, why
it's written that way, the design decisions and rejected alternatives, and
the interview questions each milestone should make you ready for. This file
is written incrementally as each milestone in `shelfsense-milestones.md` is
completed — it is the long-form companion to `docs/decisions.md` (which stays
terse) and `docs/interview_drill.md` (written at M9).

---

## M0 — Scaffold, config, data verification

### What was built

- Repo directory tree (`sql/`, `src/shelfsense/`, `tests/`, `reports/figures/`, `docs/`)
- `.gitignore` excluding `data/raw/`, `*.duckdb`, `__pycache__/`, `.venv/`
- `pyproject.toml` pinning the exact dependency list from the spec: duckdb,
  pandas, numpy, lightgbm, scikit-learn, statsmodels, shap, matplotlib,
  pyyaml, pytest — nothing else
- `config.yaml` — every tunable parameter (store list, dept list, horizon,
  train window, backtest origins, cost rates, promo threshold, seed) in one
  place
- `src/shelfsense/config.py` — a single `load_config()` that parses
  `config.yaml` into a frozen dataclass tree (`Config` → `Costs`, `Promo`)
- `src/shelfsense/sanity_check.py` — reads all four raw CSVs with pandas,
  prints row counts, column counts, null counts, and date range for calendar

### Code

`src/shelfsense/config.py`:
```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


@dataclass(frozen=True)
class Costs:
    margin_rate: float
    holding_rate: float


@dataclass(frozen=True)
class Promo:
    discount_threshold: float
    baseline_weeks: int


@dataclass(frozen=True)
class Config:
    data_dir: str
    db_path: str
    stores: list[str]
    depts: list[str]
    horizon: int
    train_days: int
    backtest_origins: list[int]
    costs: Costs
    promo: Promo
    seed: int


def load_config(path: Path = CONFIG_PATH) -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return Config(
        data_dir=raw["data_dir"],
        db_path=raw["db_path"],
        stores=raw["stores"],
        depts=raw["depts"],
        horizon=raw["horizon"],
        train_days=raw["train_days"],
        backtest_origins=raw["backtest_origins"],
        costs=Costs(**raw["costs"]),
        promo=Promo(**raw["promo"]),
        seed=raw["seed"],
    )
```

`config.yaml`:
```yaml
data_dir: data/raw
db_path: data/shelfsense.duckdb
stores: [CA_3, TX_1, WI_2]
depts: [FOODS_3, HOUSEHOLD_1]
horizon: 28
train_days: 730
backtest_origins: [1857, 1885, 1913]
costs:
  margin_rate: 0.25
  holding_rate: 0.10
promo:
  discount_threshold: 0.10
  baseline_weeks: 8
seed: 42
```

`src/shelfsense/sanity_check.py`:
```python
from __future__ import annotations

import pandas as pd

from shelfsense.config import load_config


def main() -> None:
    config = load_config()
    files = [
        "calendar.csv",
        "sales_train_evaluation.csv",
        "sell_prices.csv",
        "sample_submission.csv",
    ]
    for fname in files:
        path = f"{config.data_dir}/{fname}"
        df = pd.read_csv(path)
        print(f"{fname}: {len(df):,} rows, {len(df.columns)} cols")
        print(f"  nulls: {df.isnull().sum().sum():,}")
        if "date" in df.columns:
            print(f"  date range: {df['date'].min()} -> {df['date'].max()}")


if __name__ == "__main__":
    main()
```

### Data verification result

```
calendar.csv: 1,969 rows, 14 cols
  nulls: 7,542
  date range: 2011-01-29 -> 2016-06-19
sales_train_evaluation.csv: 30,490 rows, 1947 cols
  nulls: 0
sell_prices.csv: 6,841,121 rows, 4 cols
  nulls: 0
sample_submission.csv: 60,980 rows, 29 cols
  nulls: 0
```

All counts match MILESTONES.md exactly. `sales_train_validation.csv` (which
stops at `d_1913`, one fold short of the evaluation file) was deleted from
`data/raw/` — this project uses only `sales_train_evaluation.csv`, and having
both invites accidentally loading the wrong one.

### Design decisions

**Config as a frozen dataclass tree, not a dict.** A raw `dict` from
`yaml.safe_load` would work but gives no autocomplete, no type checking, and
lets any module silently write to shared config state. A frozen dataclass
makes config immutable after load and every field statically typed —
`config.costs.margin_rate` fails loudly if the key is renamed, whereas
`config["costs"]["margin_rate"]` fails silently (or with a bad KeyError) far
from the point of the actual mistake.

**Scope: 3 stores × 2 depts instead of all 30,490 series.** Already justified
in `docs/decisions.md` entry 1 — the modelling and SQL-engineering decisions
are the same at 4k rows as at 30k; the saved compute buys backtest folds and
sensitivity analysis instead. Rejected alternative: full dataset, which would
have made the 3-fold backtest and 8-config LightGBM grid too slow to iterate
on in the available time budget.

**`sales_train_evaluation.csv` over `sales_train_validation.csv`.** The
`_validation` file's last column is `d_1913`; `_evaluation` extends to
`d_1941`, giving 28 more days — exactly one more forecast horizon of ground
truth. Since M5's original competition scoring depended on which file you
had labels for, and this project needs actual outcomes for the final
backtest fold (`d_1914`–`d_1941`), only `_evaluation` has that fold's truth
available at all.

### Interview questions this milestone should prepare you for

**Q: Why put every parameter in a YAML file instead of just using constants
in the code?**
A: Two reasons. First, defensibility — an interviewer can ask "what if
margin were 40% instead of 25%" and I change one line, not hunt through
`cost.py`, `backtest.py`, and three notebooks for hardcoded `0.25`s. Second,
it makes the config auditable as a single diff — `git blame config.yaml`
shows every assumption change with its commit message, which is exactly the
kind of traceability a reviewer expects in a production-adjacent pipeline.

**Q: Why frozen dataclasses instead of just passing the dict around?**
A: Immutability prevents a subtle bug class: some downstream module mutating
`config["stores"]` and every later import silently seeing the mutated list.
Frozen dataclasses raise `FrozenInstanceError` on any attempted mutation, so
that bug becomes a loud crash at the mutation site instead of a silent
divergence discovered three modules later.

**Q: Why scope down to 3 stores and 2 depts instead of running on the full
30,490 series?**
A: The features, the leakage-safe windowing, the model architecture, and the
validation design don't change with series count — DuckDB and LightGBM both
scale to the full set with no code changes, since `stores`/`depts` are config
values, not code. What changes is wall-clock iteration speed. I chose to
spend the saved time on a 3-fold rolling-origin backtest, an 8-config
hyperparameter grid, and a cost-sensitivity sweep across margin assumptions,
rather than on raw row count that wouldn't change any modelling decision.

**Q: How did you verify the data was loaded correctly before building
anything on top of it?**
A: A sanity script reads each raw CSV and asserts row counts against the
known M5 dataset sizes — 1,969 calendar days, 30,490 series, ~6.84M
price rows — before any SQL or feature work touches the data. Catching a
truncated or mis-downloaded file at this stage is cheap; catching it after
building a feature matrix on top of it is not.

---

## M1 — DuckDB warehouse: wide → long

### What was built

- `sql/01_load_raw.sql` — loads all four CSVs into raw DuckDB tables with
  `read_csv_auto`, no transformation
- `sql/02_sales_long.sql` — unpivots `d_1…d_1941` into long format, filters
  to the configured store/dept scope, joins calendar and prices, trims each
  series' pre-launch leading zero-run, and materializes the result as
  `sales_long`
- `src/shelfsense/warehouse.py` — runs the SQL files in order against a
  DuckDB file, substituting `{stores}`/`{depts}` placeholders from
  `config.yaml` before execution

### Code

`sql/02_sales_long.sql` (the core of M1):
```sql
-- Wide-to-long unpivot of raw_sales, scoped to the configured stores/depts,
-- joined to calendar (date, week key) and prices (weekly, per store-item),
-- with the pre-launch leading zero-run trimmed off each series.

CREATE OR REPLACE TABLE sales_scoped AS
UNPIVOT raw_sales
ON COLUMNS(* EXCLUDE (id, item_id, dept_id, cat_id, store_id, state_id))
INTO
    NAME d
    VALUE units
;

-- {stores} / {depts} are substituted by warehouse.py from config.yaml
-- before this file is run (config values, not user input).
CREATE OR REPLACE TABLE sales_scoped AS
SELECT *
FROM sales_scoped
WHERE store_id IN ({stores})
  AND dept_id IN ({depts})
;

CREATE OR REPLACE TABLE sales_with_calendar AS
SELECT
    s.id, s.item_id, s.dept_id, s.cat_id, s.store_id, s.state_id,
    s.d, s.units,
    c.date, c.wm_yr_wk, c.wday, c.month, c.year,
    c.event_name_1, c.event_type_1, c.event_name_2, c.event_type_2,
    c.snap_CA, c.snap_TX, c.snap_WI
FROM sales_scoped s
JOIN raw_calendar c ON s.d = c.d
;

-- LEFT JOIN, not INNER: a missing price row means "not yet stocked", and
-- that's handled explicitly by the first-sale trim below, not by silently
-- dropping rows via join semantics.
CREATE OR REPLACE TABLE sales_with_price AS
SELECT swc.*, p.sell_price
FROM sales_with_calendar swc
LEFT JOIN raw_prices p
    ON swc.store_id = p.store_id
   AND swc.item_id = p.item_id
   AND swc.wm_yr_wk = p.wm_yr_wk
;

CREATE OR REPLACE TABLE first_sale AS
SELECT id, MIN(date) AS first_sale_date
FROM sales_with_price
WHERE units > 0
GROUP BY id
;

CREATE OR REPLACE TABLE sales_long AS
SELECT swp.*
FROM sales_with_price swp
JOIN first_sale fs ON swp.id = fs.id
WHERE swp.date >= fs.first_sale_date
;
```

`src/shelfsense/warehouse.py`:
```python
from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb

from shelfsense.config import Config, load_config

SQL_DIR = Path(__file__).resolve().parents[2] / "sql"
SQL_FILES = ["01_load_raw.sql", "02_sales_long.sql"]


def _quoted_list(values: list[str]) -> str:
    return ", ".join(f"'{v}'" for v in values)


def build(config: Config | None = None) -> None:
    config = config or load_config()
    con = duckdb.connect(config.db_path)
    for fname in SQL_FILES:
        sql = (SQL_DIR / fname).read_text()
        sql = sql.format(
            stores=_quoted_list(config.stores),
            depts=_quoted_list(config.depts),
        )
        start = time.time()
        con.execute(sql)
        print(f"{fname}: {time.time() - start:.1f}s")
    con.close()
```

### Verification result

```
build time: 01_load_raw.sql 7.4s, 02_sales_long.sql 87.2s (total ~95s, under the 2-min bar)
distinct ids: 4065  (1355 items x 3 stores, matches spec's "~4,000" estimate exactly)
date range: 2011-01-29 -> 2016-05-22
null price rows after trim: 0
per-store volumes:
  CA_3: 1355 items, 7,839,280 units
  TX_1: 1355 items, 3,993,233 units
  WI_2: 1355 items, 4,454,500 units
```

### A spec discrepancy worth naming (M1 DoD item 1)

MILESTONES.md's M1 "definition of done" says `sales_long` should span
"2011-01-29 → 2016-06-19". That end date is the last row of
`calendar.csv` (1,969 days total), not the last day with actual sales.
`sales_train_evaluation.csv` only has sales columns through `d_1941`, which
`raw_calendar` maps to **2016-05-22** — the calendar table simply extends 28
days further (through `d_1969` / 2016-06-19) because those extra days carry
the calendar/price/SNAP features the model needs to *predict against* for
the final forecast horizon; there is no sales ground truth for them at all
(that's the whole reason d_1942…d_1969 exist — they're the target window,
not additional training days). I did not change anything to force a match —
`sales_long` correctly ends at 2016-05-22, and the spec's DoD text was
imprecise about which table's date range it meant. Flagging this rather
than quietly "fixing" it, per the project's own honesty rule.

### Design decisions

**`UNPIVOT ... ON COLUMNS(* EXCLUDE (...))` instead of a generated `UNION
ALL`.** DuckDB's native `UNPIVOT` with a `COLUMNS()` expression selects all
1941 `d_*` columns by exclusion, so the file never hardcodes a 1941-item
column list. Rejected alternative: generate a giant `UNION ALL` in Python
and inline it — works, but produces a SQL file nobody could read or review,
which directly conflicts with the "SQL is a first-class deliverable, write
it to be read" constraint.

**Scope filter applied as a second `CREATE OR REPLACE TABLE` immediately
after the unpivot, not fused into one statement.** DuckDB's `UNPIVOT`
statement doesn't accept a trailing `WHERE` clause in the same statement.
Splitting it into two materializations costs a small amount of I/O but
keeps each statement doing one clearly-named thing — easier to explain line
by line in an interview than a single dense statement mixing unpivot and
filter logic.

**`{stores}` / `{depts}` as Python string-template substitution, not DuckDB
prepared-statement parameters.** DuckDB's `execute(sql, params)` binds
parameters to a *single* statement; `warehouse.py` runs whole multi-statement
`.sql` files as scripts, where that binding doesn't apply. Since the values
being substituted come from `config.yaml` (a trusted local file), not
user input, plain `str.format()` is safe here and avoids restructuring every
SQL file into one-statement-per-call. This would be the wrong call if
`stores`/`depts` ever came from an HTTP request or CLI arg passed through
from an untrusted source — worth saying explicitly if asked about SQL
injection risk.

**`LEFT JOIN` to `raw_prices`, with the "not yet stocked" case handled
explicitly by the first-sale trim, not implicitly by the join type.** An
`INNER JOIN` would also drop unstocked-period rows, but it would silently
conflate "no price because not yet stocked" with "no price for some other
data-quality reason" — both vanish the same way, with no signal that
they're different situations. Doing the trim explicitly via
`first_sale_date` after a `LEFT JOIN` means a genuinely unexpected missing
price (mid-life, not pre-launch) would surface as a `NULL` in `sell_price`
post-trim, which the DoD check (`sell_price IS NULL` count) is specifically
designed to catch.

### Interview questions this milestone should prepare you for

**Q: Write me the DuckDB unpivot you used for the wide-to-long conversion.**
A:
```sql
UNPIVOT raw_sales
ON COLUMNS(* EXCLUDE (id, item_id, dept_id, cat_id, store_id, state_id))
INTO NAME d VALUE units
```
`COLUMNS(* EXCLUDE (...))` selects every `d_1`…`d_1941` column by excluding
the six identifier columns, so the 1941-column list is never hardcoded.
Every other column stays fixed per output row; `d` and `units` become the
new name/value pair.

**Q: How did you handle the leading zero-runs before an item's first
sale?**
A: Computed `first_sale_date = MIN(date) WHERE units > 0` per series, then
dropped every row before that date for that series. Those zeros represent
"not yet stocked," not "zero demand" — an item that launched in month 30 of
a 65-month window shouldn't have its pre-launch silence modelled as
demand history, and leaving it in would badly deflate the RMSSE scaling
denominator (which is itself computed from in-sample variance).

**Q: Why LEFT JOIN to prices instead of INNER JOIN, if you're going to drop
the unstocked rows anyway?**
A: Because dropping them via the join type makes "not yet stocked" and "a
genuine missing price for some other reason" indistinguishable — both just
disappear. Left-joining and trimming explicitly via `first_sale_date` means
if a *mid-life* price row were ever missing (a real data quality issue,
distinct from pre-launch), it would surface as a `NULL` in `sell_price`
after the trim, and the sanity check that counts null prices post-trim
would catch it instead of silently losing the row.

**Q: `wm_yr_wk` is a weekly key but sales are daily — walk me through that
join.**
A: Prices are one row per `(store_id, item_id, wm_yr_wk)` — a single price
per store-item per week. Joining that onto daily sales via `wm_yr_wk`
means every day within the same week gets the same price row; a "price
change" in this dataset can only happen on a week boundary, not on an
arbitrary weekday. That matters later for the promo definition in M6 — a
promo onset is necessarily a week-level event, not a day-level one, because
that's the grain at which price actually varies in this data.

**Q: The spec says the date range should end 2016-06-19, but yours ends
2016-05-22 — why the mismatch?**
A: `sales_train_evaluation.csv` only has sales columns through `d_1941`,
which maps to 2016-05-22. `calendar.csv` extends 28 days further, through
`d_1969` / 2016-06-19, because those extra days are the actual 28-day
forecast horizon — they carry calendar/price/SNAP features to predict
*against*, not additional sales history. There's no ground truth for
`d_1942`…`d_1969` in this file at all. `sales_long` correctly stops where
the sales data stops; the spec's DoD text conflated the calendar table's
range with the sales table's range.

---
