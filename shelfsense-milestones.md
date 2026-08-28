# shelfsense — Milestone Build Doc (v2, scoped)

**Built for:** Tredence Analyst / Associate Data Scientist (2 technical rounds), and reusable for any Analyst / Data Scientist / ML role that asks for Python + advanced SQL + statistical modelling + interpretable models + business impact.

**Budget:** ~30 focused hours. 5 days at 6 hrs/day, or 4 days at 7.5.

**One-line pitch:**
> A retail demand forecasting and promotional-uplift system on Walmart M5 data that answers two questions a category manager actually asks — *how much will this store-item sell over the next 28 days*, and *did that promotion actually cause the lift, or would it have sold anyway* — and converts both answers into a dollar cost-of-error number.

---

## 0. What changed from v1 (and why this matters in the interview)

v1 was a 6-day, M0–M15 build that tried to cover forecasting **and** uplift **and** deep learning **and** an app layer. That is a portfolio project that gets 40% built and can't be defended end to end.

v2 cuts scope on **breadth** and spends the saved time on **depth of defence**. Every cut below is a deliberate answer you give when an interviewer asks "why didn't you do X".

| Cut | Kept instead | Your answer when asked |
|---|---|---|
| All 30,490 series | 3 stores × 2 departments ≈ 4,000 series | "The modelling decisions are identical at 4k and 30k series. I chose to spend compute on validation rigour rather than row count, and the pipeline is parameterised — changing the store filter is one config line." |
| LSTM / N-BEATS / Prophet | Seasonal-naive baselines + one global LightGBM | "The JD asks for interpretable models. Gradient boosting on engineered features beat the baselines by X% and I can explain every feature to a category manager. A neural forecaster would have bought me maybe 1–2% for zero explainability." |
| MLOps, Docker, cloud, FastAPI | A clean, tested, reproducible repo | "This is an analytics deliverable, not a service. Tredence Studio ships white-box solutions on someone else's platform — the value is in the model and the recommendation, not in me hosting it." |
| 12-level WRMSSE | RMSSE at series level, dollar-weighted | "I implemented the scaled error properly and weighted by revenue. Full 12-level aggregation is a Kaggle-leaderboard artifact, not a business one." |
| Streamlit dashboard | A 1-page business memo + 5 charts | "The audience for this is an exec. A memo with a dollar number lands better than an app nobody opens." |

**Rule for the whole build: if something doesn't end up either (a) in the results memo or (b) on the interview drill sheet, don't build it.**

---

## 1. Scope lock

### Dataset
Walmart M5 (Kaggle: *M5 Forecasting – Accuracy*). Four files:
- `calendar.csv` — 1,969 days, 2011-01-29 → 2016-06-19, with `wm_yr_wk`, `event_name_1/2`, `event_type_1/2`, `snap_CA`, `snap_TX`, `snap_WI`
- `sales_train_evaluation.csv` — 30,490 rows × `d_1 … d_1941` (use this one, **not** `sales_train_validation.csv`, which stops at `d_1913`)
- `sell_prices.csv` — weekly price per `store_id × item_id × wm_yr_wk`; null before the item was first stocked
- `sample_submission.csv` — ignore, you're not submitting

### The slice
```
stores   : CA_3, TX_1, WI_2        (one per state — needed for the SNAP robustness check)
depts    : FOODS_3, HOUSEHOLD_1    (the two highest-volume departments)
days     : d_1 … d_1941, but model training uses the last 730 days only
```
≈ 4,000 series × 1,941 days ≈ 7.8M rows in DuckDB, ≈ 2.9M rows in the training matrix. Trains in single-digit minutes on a laptop.

### The two questions
1. **Forecast.** 28-day-ahead daily unit demand per store-item.
2. **Uplift.** For a price promotion (defined below), what was the causal lift in units, and was it profitable?

### The business frame (this is the "now what" the JD keeps asking for)
- Understock cost `c_u` = lost gross margin per unit = `price × margin_rate` (assume 25%)
- Overstock cost `c_o` = holding + markdown per unit = `price × 0.10`
- Critical ratio = `c_u / (c_u + c_o)` = 0.25/(0.25+0.10) ≈ **0.714**
- So the correct order quantity is the **71.4th percentile** of the demand distribution, not the mean. This single insight is the spine of the whole project.

---

## 2. Repo shape

```
shelfsense/
├── README.md                    # written LAST, at M8
├── pyproject.toml
├── .gitignore                   # data/raw/ must be ignored
├── config.yaml                  # stores, depts, horizon, cost params, seeds
├── data/
│   ├── raw/                     # the 4 Kaggle CSVs
│   └── shelfsense.duckdb        # built artifact, gitignored
├── sql/
│   ├── 01_load_raw.sql
│   ├── 02_sales_long.sql
│   ├── 03_calendar_features.sql
│   ├── 04_price_features.sql
│   ├── 05_feature_matrix.sql
│   └── 06_promo_panel.sql
├── src/shelfsense/
│   ├── config.py                # loads config.yaml, single source of truth
│   ├── warehouse.py             # runs the sql/ files in order
│   ├── metrics.py               # rmsse, weighted_rmsse, mae, bias, pinball
│   ├── baselines.py
│   ├── model.py                 # LightGBM train/predict, mean + quantile
│   ├── backtest.py              # rolling-origin harness
│   ├── cost.py                  # newsvendor / cost-of-error simulation
│   ├── uplift.py                # DiD, event study, power analysis
│   ├── explain.py               # SHAP
│   └── plots.py
├── tests/
│   ├── test_leakage.py          # the important one
│   ├── test_metrics.py
│   └── test_splits.py
├── reports/
│   ├── figures/
│   ├── results.md               # full technical write-up
│   └── one_pager.md             # the exec memo
└── docs/
    ├── decisions.md             # running log of every choice + rejected alternative
    └── interview_drill.md       # Q&A you rehearse from
```

**Environment:** Python 3.11, `uv` or plain venv. Pinned: `duckdb`, `pandas`, `numpy`, `lightgbm`, `scikit-learn`, `statsmodels`, `shap`, `matplotlib`, `pyyaml`, `pytest`. Nothing else. If you find yourself adding a dependency, that's scope creep.

---

## M0 — Scaffold, data, and scope lock
**Time: 2 hrs** · **Day 1 morning**

### Goal
Repo exists, data is on disk and verified, config drives everything, first commit is clean.

### Steps
1. `git init`, create the directory tree above, write `.gitignore` (`data/raw/`, `*.duckdb`, `__pycache__/`, `.venv/`).
2. Download the 4 M5 CSVs into `data/raw/`.
3. Write `config.yaml`:
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
     discount_threshold: 0.10   # >=10% below trailing median price
     baseline_weeks: 8
   seed: 42
   ```
4. Write `src/shelfsense/config.py` — one `load_config()` returning a dataclass. **No magic numbers anywhere else in the codebase.**
5. Write a 10-line sanity script that prints row counts, date range, and null counts for each raw file. Save output to `docs/decisions.md` as the first entry.

### Definition of done
- [ ] `python -m shelfsense.config` prints the parsed config
- [ ] Sanity script confirms: calendar 1,969 rows; sales_train_evaluation 30,490 rows; sell_prices ~6.84M rows
- [ ] First commit pushed, README is a stub with just the one-line pitch
- [ ] `docs/decisions.md` has entry 1: "Scoped to 3 stores × 2 depts. Reason: …"

### Gotcha
Do **not** load M5 with pandas `read_csv` on the wide file and then melt in pandas — it's slow and you'll fight memory. DuckDB reads the CSV directly. That's M1.

---

## M1 — DuckDB warehouse: wide → long
**Time: 3 hrs** · **Day 1 afternoon**

### Goal
A single `.duckdb` file with a tidy long-format sales table, filtered to the scope, joined to calendar and prices.

### Why this milestone matters for Tredence
The JD names **advanced SQL** as a required skill and it's the one thing most portfolio ML projects skip entirely (everyone does pandas). Doing the entire feature layer in SQL — CTEs, window functions, `QUALIFY`, frame clauses — is the single highest-leverage differentiator in this build. You want an interviewer to open `sql/05_feature_matrix.sql` and see real SQL.

### Steps
1. `sql/01_load_raw.sql` — `CREATE TABLE raw_calendar AS SELECT * FROM read_csv_auto(...)` for all four files.
2. `sql/02_sales_long.sql` — unpivot `d_1 … d_1941` into `(id, item_id, dept_id, cat_id, store_id, state_id, d, units)`.
   - DuckDB has `UNPIVOT`; use it. Fallback: generate the `UNION ALL` SQL programmatically.
   - Filter to scope in the same statement: `WHERE store_id IN (...) AND dept_id IN (...)`.
3. Join `d` → `date`, `wm_yr_wk` from `raw_calendar`.
4. Join `sell_prices` on `(store_id, item_id, wm_yr_wk)`.
5. Add `first_sale_date` per series and **drop all rows before it** — M5 series have long leading zero-runs from before the item was stocked. Modelling those as demand is wrong and will wreck your scaled error denominator.
6. Materialise as table `sales_long`.

### Definition of done
- [ ] `sales_long` exists, ~4,000 distinct `id`, dates 2011-01-29 → 2016-06-19
- [ ] Zero rows where `sell_price IS NULL` after the first-sale trim (spot-check; a handful is fine, log them)
- [ ] Query `SELECT store_id, COUNT(DISTINCT id), SUM(units) FROM sales_long GROUP BY 1` returns sensible per-store volumes
- [ ] `warehouse.py build` rebuilds the whole DB from scratch in under 2 minutes

### Gotcha
`wm_yr_wk` is a **week** key. Prices are weekly, sales are daily. The join fans out correctly, but be aware that a "price change" happens on week boundaries, not arbitrary days. Your promo definition in M6 must respect this.

---

## M2 — SQL feature layer
**Time: 4 hrs** · **Day 2 morning**

### Goal
A single wide `feature_matrix` table, built entirely in SQL, with every feature explicitly labelled as **known-at-forecast-time** or **not**.

### The leakage rule (write this in a comment at the top of `05_feature_matrix.sql`)
> You are forecasting 28 days ahead. On the forecast date `t`, you know sales up to `t-1` only. Therefore every lag and rolling feature must be shifted by at least `h=28`. Price, calendar, SNAP and event features are *plans*, known in advance, and may be used at time `t` directly.

Getting this distinction right and being able to articulate it is worth more in an interview than any modelling choice in the project.

### Features to build

**Lag features** (all shifted ≥ 28):
- `lag_28`, `lag_35`, `lag_42`, `lag_56`, `lag_364` (same day last year)

**Rolling features** (window ends at `t-28`):
- rolling mean of units over 7, 28, 56, 180 days
- rolling std over 28, 56 days
- rolling max over 28 days
- rolling share-of-zero-days over 28 days (intermittency signal)
- current zero-run length as of `t-28`

Implement with `AVG(units) OVER (PARTITION BY id ORDER BY date ROWS BETWEEN 55 PRECEDING AND 28 PRECEDING)` — the frame clause **is** the shift. Don't lag then roll; roll with an offset frame.

**Price features** (known in advance, no shift needed):
- `sell_price`
- `price_vs_item_median_8w` — price ÷ trailing 8-week median price for that store-item
- `price_vs_dept_median_week` — price ÷ median price in that dept-store that week
- `price_changed_flag` — price differs from previous week
- `price_momentum_4w`

**Calendar features** (known in advance):
- `dow`, `day_of_month`, `week_of_year`, `month`, `year`
- `is_weekend`
- `snap` — the SNAP flag *for that row's own state* (pick the right column per state; do it with a `CASE`)
- `event_type_1` as a low-cardinality categorical, plus `days_to_next_event` and `days_since_last_event`
- `is_christmas` — Walmart is closed Dec 25, sales are 0. Flag it; don't let the model learn it as demand collapse.

**Identity features** (categorical, let LightGBM handle natively):
- `item_id`, `dept_id`, `cat_id`, `store_id`, `state_id`

**Series meta:**
- `days_since_first_sale`

### Definition of done
- [ ] `feature_matrix` builds from `sales_long` in one SQL file, under 90 seconds
- [ ] `tests/test_leakage.py` passes: for a random sample of 200 rows, assert every `lag_*` / `roll_*` value can be reconstructed from `sales_long` using only `date <= t - 28`
- [ ] A markdown table in `docs/decisions.md` listing every feature with a `known_at_forecast_time: yes/no` column
- [ ] Null rate per feature logged; nulls only where expected (early rows)

### Gotcha
`lag_364` will be null for the first year of every series. That's fine — LightGBM handles nulls natively. Do **not** impute with zero; zero is a meaningful value in this dataset and imputing destroys the signal.

---

## M3 — Metrics + baselines + backtest harness
**Time: 3 hrs** · **Day 2 afternoon**

### Goal
You can score any forecast, and you have numbers to beat.

### Metrics (`metrics.py`)
- **RMSSE** per series: `sqrt( mean((y - yhat)^2) / mean(diff(y_train)^2) )`. The denominator is the in-sample one-step naive error — it's what makes the error *scaled* and comparable across a 2-units/day item and a 200-units/day item.
- **Weighted RMSSE**: weight each series by its dollar sales in the last 28 days of the training window. This is the business-relevant aggregate.
- **MAE**, **bias** (mean signed error) — bias matters enormously for inventory and nobody reports it.
- **Pinball loss** at a given quantile — needed in M5.

### Baselines (`baselines.py`)
1. `naive` — last observed value
2. `seasonal_naive_7` — value 7 days ago (i.e. same weekday last week), taken from `t-28` so it's horizon-legal
3. `mean_28` — trailing 28-day mean
4. `croston_lite` — for intermittent series, mean of non-zero values × non-zero rate

Baseline 2 or 3 will be surprisingly hard to beat. Say so in the writeup; it's a sign of honesty.

### Backtest harness (`backtest.py`)
Rolling origin, three folds, each a 28-day block:

| Fold | Train through | Predict |
|---|---|---|
| 1 | `d_1857` | `d_1858 – d_1885` |
| 2 | `d_1885` | `d_1886 – d_1913` |
| 3 | `d_1913` | `d_1914 – d_1941` |

Train window = trailing 730 days from each origin.

### Definition of done
- [ ] `python -m shelfsense.backtest --model seasonal_naive_7` runs all 3 folds and prints a per-fold + mean metrics table
- [ ] `tests/test_metrics.py`: RMSSE of a perfect forecast = 0; RMSSE of the naive forecast on the training tail ≈ 1
- [ ] `tests/test_splits.py`: no train index ever has `date >= ` any test date in the same fold
- [ ] Baseline table saved to `reports/results.md`

### Gotcha
Compute the RMSSE denominator from the **training** window of that fold only. Using the full history leaks and makes your scores look better than they are.

---

## M4 — LightGBM global model
**Time: 4 hrs** · **Day 3 morning**

### Goal
One global model across all series that beats every baseline on weighted RMSSE.

### Design
- **One model, all series** (a "global" model), not one model per series. Argue this: 4,000 individual models can't share the SNAP effect or the Thanksgiving effect; a global model learns them once from 4,000 series' worth of evidence, and handles short-history items.
- **Direct multi-horizon**, not recursive. Train one model that predicts `units` at any offset within the 28-day window, with `days_ahead` as a feature. Recursive prediction compounds error over 28 steps and is a common failure people don't defend well.
- Objective: `tweedie` (variance_power ≈ 1.1). Retail unit demand is non-negative, zero-inflated, right-skewed. Tweedie is the standard choice and saying *why* is an easy interview win. Compare against `poisson` and `l2` in one experiment and report the difference.
- Categoricals: pass `item_id`, `dept_id`, `store_id`, `state_id`, `event_type_1` as native LightGBM categoricals.

### Tuning — deliberately light
Do **not** run a 200-trial Optuna study. Run a hand-specified grid of ~8 configurations over `num_leaves ∈ {64, 128, 256}`, `learning_rate ∈ {0.05, 0.1}`, `min_data_in_leaf ∈ {50, 200}`, with early stopping on fold 1. Pick the best, use it for folds 2 and 3. Log all 8 results.

Your line: "I capped tuning at 8 configurations because the gap between the best and median config was under 2% weighted RMSSE, while the gap between the baseline and any LightGBM config was over 20%. Feature quality was the binding constraint, not hyperparameters." Have the numbers to back that.

### Definition of done
- [ ] Beats `seasonal_naive_7` by a meaningful margin on weighted RMSSE across all 3 folds (target: 15%+ improvement; report whatever you actually get)
- [ ] Per-fold results table in `reports/results.md`
- [ ] Objective comparison (tweedie / poisson / l2) logged
- [ ] Training run reproducible from a fixed seed; model artifact saved
- [ ] Bias reported — if the model is systematically under-forecasting, say so and explain what that costs

---

## M5 — Quantile forecasts and the cost-of-error simulation
**Time: 3 hrs** · **Day 3 afternoon**

### Goal
Convert forecast accuracy into money. **This is the milestone that makes the project a Tredence project rather than a Kaggle project.**

### Steps
1. Retrain the same LightGBM config with `objective='quantile'` at `alpha = 0.714` (the critical ratio from §1) and also at `alpha ∈ {0.5, 0.9, 0.95}` for the chart.
2. Build the inventory simulation in `cost.py`. For each store-item-day in the test folds:
   - `order_qty` = the policy's forecast
   - `understock = max(0, actual - order_qty)` → cost `understock × price × 0.25`
   - `overstock = max(0, order_qty - actual)` → cost `overstock × price × 0.10`
3. Compare four policies on total cost:
   - **P0**: seasonal-naive
   - **P1**: LightGBM mean forecast, no safety stock
   - **P2**: LightGBM mean forecast + safety stock at 1.65σ (the classic "95% service level" heuristic)
   - **P3**: LightGBM quantile forecast at the critical ratio
4. Report: total cost per policy, cost reduction vs P0 in %, and **absolute dollars over the 84 test days across the 4,000 series**. Then extrapolate to the full 30,490-series chain and state the extrapolation assumption explicitly.
5. Sensitivity: re-run for `margin_rate ∈ {0.15, 0.25, 0.40}`. Show that the *ranking* of policies is stable even though the absolute savings move. This pre-empts "but you made those cost numbers up".

### Definition of done
- [ ] Cost table for all 4 policies × 3 margin assumptions
- [ ] Chart: service level (x) vs total cost (y), with the newsvendor optimum marked — this is your best single slide
- [ ] A one-sentence finding of the form: "Ordering to the 71st percentile instead of the mean cuts total cost of error by X%, worth approximately $Y over 28 days on this slice."
- [ ] Sensitivity results logged in `docs/decisions.md`

### Interview line this earns
> "Most forecasting projects stop at MAE. But minimising MAE is optimal only if over- and under-forecasting cost the same, and in retail they never do — a stockout costs you the margin, an overstock costs you the holding. Once I wrote down the two costs, the correct thing to forecast stopped being the mean and became the 71st percentile. That reframing was worth more than any model improvement I made."

---

## M6 — Promotional uplift: DiD, event study, power analysis
**Time: 4 hrs** · **Day 4 morning**

### Goal
Answer *did the promo cause the lift* with a defensible causal design, and quantify what effect size you could even have detected.

### Why this milestone
The JD explicitly names **hypothesis testing**, **sample size estimation**, and **A/B testing**. This milestone is the direct answer to all three. It's also the part that separates a data scientist from someone who fits models.

### Define the promotion
A store-item-week is **treated** if:
```
sell_price <= 0.90 × (trailing 8-week median sell_price for that store-item)
```
and the preceding 4 weeks were not themselves treated (so you isolate promo *onsets*, not sustained price cuts).

Build `sql/06_promo_panel.sql` producing an event-time panel: for each promo onset, weeks `-4 … +4` relative to onset, with log units, price, and store/item/week identifiers.

### Design 1 — Difference-in-differences (primary)
- **Treated**: store-item-weeks with a promo onset
- **Control**: store-item-weeks in the *same store and department, same calendar week*, with no promo and no promo in the surrounding 4 weeks
- Model: `log1p(units) ~ treated × post + item_store_FE + week_FE`, estimated with `statsmodels` OLS, **standard errors clustered at the store-item level**
- Report the coefficient on `treated × post` as a % lift with a 95% CI

### Design 2 — Event study (the parallel-trends check)
Same regression but with separate coefficients for each relative week `-4 … +4`, omitting `-1` as reference. Plot them with CIs.
- If the pre-period coefficients (`-4 … -2`) are flat and near zero → parallel trends is plausible, and you can say so with a figure rather than an assertion
- If they trend up → **say so honestly**. That's the anticipation/endogeneity problem: Walmart discounts items it already expects to move. Naming this threat unprompted is the strongest thing you can do in the interview.

### Design 3 — Placebo / falsification
Re-run the DiD with fake promo dates shifted 8 weeks earlier. The estimated "effect" should be indistinguishable from zero. If it isn't, your design is picking up seasonality, not promotion.

### Power analysis (sample size estimation)
Using the observed within-series standard deviation of `log1p(units)`:
- For the treated sample size you actually have, compute the **minimum detectable effect** at 80% power, α = 0.05
- Then invert it: how many treated store-item-weeks would you need to detect a 5% lift? A 2% lift?
- Present as a small table: MDE vs N, and mark where your study sits

State the conclusion plainly: *"With N treated units I could reliably detect a lift of X% or larger. The effect I estimated was Y%, so the study was adequately powered / underpowered for this question."*

### The A/B design section (~200 words in `results.md`)
Write out how you'd run this properly if you could randomise: randomisation unit (store-item, not store — explain why), treatment assignment, minimum runtime from the power calc, guardrail metrics (category cannibalisation, margin not just units), and the pre-registered primary metric. This costs you 20 minutes and directly answers a JD bullet.

### Definition of done
- [ ] DiD point estimate + clustered 95% CI
- [ ] Event-study plot saved to `reports/figures/`
- [ ] Placebo test result reported
- [ ] MDE table + a stated power conclusion
- [ ] A written **limitations** paragraph naming: non-random assignment, price endogeneity, cannibalisation not measured, weekly price granularity
- [ ] The A/B design section written

### Gotcha
Do not use the same data to both select promos and estimate their effect without noting it. And do not report a p-value without the effect size and CI next to it — that's the single most common tell of someone who learned stats from a tutorial.

---

## M7 — Interpretability and error diagnostics
**Time: 2 hrs** · **Day 4 afternoon**

### Goal
Explain the model to a non-technical stakeholder, and know where it fails.

### Steps
1. **SHAP global**: `TreeExplainer` on a 50k-row sample of the test fold. Beeswarm plot + mean-|SHAP| bar chart.
2. **SHAP local**: two worked cases — one large over-forecast and one large under-forecast. Walk through which features drove each. Put both in `results.md` as narrative, not just plots.
3. **Error segmentation**: weighted RMSSE broken out by
   - department
   - store
   - demand volume decile
   - intermittency (share of zero days)
   - SNAP day vs non-SNAP day
   - event weeks vs normal weeks
4. Write 3–5 sentences on **where the model is weakest** and what you'd do about it with another week. (Expected answer: high-intermittency, low-volume items — the model over-smooths them; you'd add a dedicated intermittent-demand treatment.)

### Definition of done
- [ ] Beeswarm + importance plots in `reports/figures/`
- [ ] Two local explanations written out in prose
- [ ] Error segmentation table
- [ ] "Where it fails" paragraph written

### Gotcha
SHAP on the full test set will be slow and you don't need it. Sample. And beware reading SHAP importance as causal — it tells you what the *model* used, not what *drives demand*. Say this out loud in the interview; it's a distinction a lot of candidates miss.

---

## M8 — Results write-up, exec one-pager, README
**Time: 3 hrs** · **Day 5 morning**

### Goal
The artifacts a recruiter, an interviewer, and an exec each actually read.

### `reports/results.md` (technical, ~1,500 words)
1. Problem and business framing
2. Data and scope, with the scoping rationale
3. Feature engineering, with the known-at-forecast-time table
4. Validation design
5. Model results: baselines → LightGBM → quantile, all three folds
6. Cost-of-error results and the newsvendor chart
7. Uplift: DiD, event study, placebo, power
8. Interpretability and error segmentation
9. Limitations (be generous here — 6+ named limitations)
10. What I'd do with another two weeks

### `reports/one_pager.md` (exec, ~250 words + 3 charts)
Structure it as a memo, not a report:
- **Recommendation** (first line — the "now what"): e.g. *"Move from mean-based to 71st-percentile ordering on FOODS_3 and HOUSEHOLD_1. Estimated cost-of-error reduction: X%."*
- **Evidence**: 3 bullets, each with a number
- **Confidence and caveats**: 2 bullets
- **What we'd need to deploy**: 2 bullets

Charts: (1) service-level vs cost curve, (2) event-study plot, (3) forecast vs actual for one representative series.

### `README.md`
- One-line pitch, then a results table above the fold — the headline numbers must be visible without scrolling
- The two charts that matter
- Setup: 3 commands, must work from a clean clone
- Repo map
- A short "design decisions" section linking to `docs/decisions.md`

### Definition of done
- [ ] Fresh clone → `uv sync` → `python -m shelfsense.warehouse build` → `python -m shelfsense.backtest --all` reproduces the headline numbers
- [ ] README shows results above the fold
- [ ] One-pager reads like something you'd actually send a client
- [ ] All figures are legible at thumbnail size (label your axes, no default matplotlib titles)

---

## M9 — Interview drill sheet
**Time: 2 hrs** · **Day 5 afternoon** · **Do not skip this milestone**

Tredence runs **two technical rounds**. The project's value is entirely in how you talk about it. Write `docs/interview_drill.md` with your actual answers — written out, not bullet-pointed intentions.

### The 20 questions to have answers for

**Framing (4)**
1. Walk me through this project in 90 seconds.
2. Who is the user of this and what decision does it change?
3. Why forecasting *and* uplift — why not just one?
4. What would you do differently with a real client and real data?

**Data & SQL (3)**
5. Why did you do feature engineering in SQL rather than pandas?
6. Write me a window function that computes a 28-day trailing average lagged by 28 days.
7. How did you handle the leading zeros in M5?

**Modelling (5)**
8. Why a global model instead of per-series models?
9. Why Tweedie loss?
10. Why direct multi-horizon instead of recursive?
11. Your baseline is close to your model on some segments — why keep the model?
12. How do you know you didn't leak?

**Statistics & causality (5)**
13. Why DiD and not just a before/after comparison?
14. How did you test parallel trends?
15. What's your identification threat, and how worried are you?
16. What was your minimum detectable effect and how did you compute it?
17. If you could randomise, how would you design the experiment?

**Business (3)**
18. Where did the 25% margin and 10% holding cost come from?
19. How sensitive is your recommendation to those assumptions?
20. What's the one number you'd put in front of the category director?

### Also prepare
- A **60-second version** and a **5-minute version** of the walkthrough
- The **three numbers you never fumble**: baseline weighted RMSSE, model weighted RMSSE, % cost reduction from the quantile policy
- One **thing that went wrong** and how you diagnosed it (interviewers ask this constantly and "nothing went wrong" is a bad answer)

### Definition of done
- [ ] All 20 answers written out in full sentences
- [ ] You can do the 90-second walkthrough out loud without reading
- [ ] The three headline numbers are memorised

---

## Resume bullets (drop into `AashikSandeep_ML`)

Replace the current shelfsense entry with three lines in this shape — fill in your real numbers:

> **shelfsense** — Retail demand forecasting & promotional uplift · *Python, DuckDB/SQL, LightGBM, statsmodels, SHAP*
> - Built a 28-day-ahead global demand forecaster over ~4,000 Walmart store-item series; engineered 30+ leakage-safe lag, rolling and price features **entirely in SQL** (window functions with offset frames), cutting weighted RMSSE **XX%** vs a seasonal-naive baseline across a 3-fold rolling-origin backtest.
> - Reframed the objective from accuracy to cost: derived the newsvendor critical ratio from stockout vs holding costs and forecast the **71st percentile instead of the mean**, reducing total cost of error **XX%** and holding that ranking across a 3-point margin sensitivity.
> - Measured promotional uplift causally with difference-in-differences on price-cut onsets (store-item and week fixed effects, clustered SEs), validated with an event-study parallel-trends check and a placebo test, and reported the **minimum detectable effect** from a power analysis alongside the estimate.

---

## Schedule

| Day | Morning (3–4h) | Afternoon (3h) |
|---|---|---|
| 1 | M0 scaffold + data | M1 DuckDB warehouse |
| 2 | M2 SQL feature layer | M3 metrics + baselines + backtest |
| 3 | M4 LightGBM | M5 cost-of-error |
| 4 | M6 uplift + power | M7 SHAP + diagnostics |
| 5 | M8 write-up + README | M9 interview drill |

**If you lose a day:** cut M7 to SHAP-importance-only (45 min) and keep M9. Never cut M5, M6, or M9 — those three are the entire differentiation.

---

## The two-sentence version to have ready

> "I built a demand forecaster for Walmart store-item data, but the part I care about is that I didn't optimise for accuracy — I wrote down what a stockout costs versus what an overstock costs, and that told me to forecast the 71st percentile rather than the mean, which cut cost of error by more than the model improvement did. Then I asked whether the promotions in the data actually caused their lift, ran a difference-in-differences with an event-study check and a placebo test, and reported the minimum detectable effect so you know what the study could and couldn't have found."
