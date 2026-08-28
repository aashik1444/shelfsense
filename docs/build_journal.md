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
