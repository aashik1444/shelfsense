# Interview drill sheet

15 questions an interviewer would ask about this project, with answers
grounded in what the code actually does — not what would sound good.
Written for the scope-frozen state at M4 (see `docs/decisions.md` Entry 6).

---

**Q1: Why RMSSE instead of MAPE?**

MAPE divides by the actual value, which is undefined or explodes on the
zero-demand days that make up a large share of this dataset — MAPE simply
cannot be computed on a day with zero units sold. RMSSE instead scales the
squared forecast error by the in-sample one-step naive error computed from
that fold's training window, so it's comparable across a 2-unit/day item
and a 200-unit/day item without blowing up on zeros. `tests/test_metrics.py`
checks this directly: RMSSE of a perfect forecast is 0, and RMSSE of the
naive forecast measured against its own training tail is ≈1 by
construction, which is the sanity check that confirms the scaling is
implemented correctly.

**Q2: The model only beats `mean_28` by 5.2% — would you still deploy it?**

I'd want to see the segment-level breakdown before answering yes or no,
and that breakdown doesn't exist yet — it was scoped into M7, which was
cut when the project froze at M4. What I can say honestly: `mean_28` beats
LightGBM's margin over `seasonal_naive_7` (27.6%) by a lot, which tells me
`mean_28` already captures most of the trend/level signal in this
particular slice — a stable, slow-moving product mix. LightGBM's edge has
to be coming from price, calendar, SNAP, and event features that `mean_28`
structurally can't use at all. If that edge concentrates in holiday weeks
or promotional periods — exactly where a 5.2% aggregate number would
understate the real business value — I'd deploy it. If the edge is spread
thin and uniform, a category manager might reasonably ask whether the
added complexity (a trained model vs. a rolling average anyone can compute
in a spreadsheet) is worth it. I don't have the segmentation to answer
that definitively, and I'd say so rather than guess.

**Q3: Walk me through the leakage shift, and why is `seasonal_naive_7`
built from `lag_35` instead of `lag_7`?**

Every lag and rolling feature in `sql/05_feature_matrix.sql` is built with
a window FRAME clause whose upper bound encodes the shift — e.g.
`AVG(units) OVER (PARTITION BY id ORDER BY date ROWS BETWEEN 55 PRECEDING
AND 28 PRECEDING)`. The frame's most recent visible row is 28 rows before
the current one, so the feature structurally cannot see anything from
`t-27` through `t`. `seasonal_naive_7` — "same weekday last week" — sounds
like it should be `lag_7`, but at forecast time `t` we only know data
through `t-1`, and this project's horizon is 28 days, so the freshest
anchor point that's horizon-legal for *any* feature is `t-28`. "One week
before the most recent legal anchor" is `t-28-7 = t-35`. Using `lag_7`
here would leak exactly the same way an unshifted rolling feature would —
it's just dressed up as a baseline instead of a model feature, which makes
it an easy mistake to miss if you're only scrutinizing the model code.

**Q4: Why Tweedie when L2 tied it in the objective comparison?**

L2 scored 0.8275 and Tweedie scored 0.8279 on fold 1 with the tuned
hyperparameters — a 0.05% gap, well within noise. I chose Tweedie anyway
because the choice of loss function should reflect the data's actual
distributional shape, not chase a within-noise difference on one
particular 3-store/2-dept slice. Retail unit demand is non-negative,
often exactly zero, and right-skewed when it isn't — Tweedie with
`variance_power` between 1 and 2 is built for exactly that shape,
interpolating between Poisson (count-like) and Gamma (continuous, skewed)
behavior. A 0.05% gap on this slice isn't evidence that L2 is the better
general choice — it's evidence the two are close enough here that
hyperparameter tuning mattered more than objective choice for this
specific data. On a different slice with heavier intermittency, I'd
expect the gap to open up in Tweedie's favor.

**Q5: Why a global model instead of one model per series?**

Two reasons. Statistically: 4,065 individual models can't share the SNAP
effect, a holiday effect, or any cross-series pattern — each has to
relearn it from that single series' history alone, which is especially
weak for short-history items. A global model learns those effects once
from all series' evidence and applies them everywhere. Practically: a
brand-new item with only a few weeks of history is handled gracefully by
a global model (it borrows structure from similar items via the
categorical and price features), where a per-series model for that item
would have almost nothing to train on.

**Q6: Why direct multi-horizon instead of recursive?**

Recursive forecasting predicts day `t+1`, feeds that prediction back in as
a lagged feature to predict `t+2`, and so on for 28 steps — so any error
in the day-1 prediction compounds through all 27 subsequent steps, and by
day 28 the model may be conditioning on values that look nothing like real
data. Direct multi-horizon trains one model to predict any offset within
the horizon directly from features known at the origin, with `days_ahead`
as a feature, so every prediction is made from real observed history, not
the model's own earlier guesses. Concretely: each anchor row in
`feature_matrix` at date `t` already has features known at `t-28`; I
reuse that one anchor to build 28 training examples — one per
`days_ahead` in [1,28] — predicting `units` at `(t-28) + days_ahead`. The
entire multi-horizon target set comes from one already-shifted feature
row, with no need to build 28 separate feature tables at 28 different
shift amounts.

**Q7: Why rolling-origin backtesting instead of k-fold cross-validation?**

K-fold cross-validation shuffles rows across folds, which for time series
means training on data from *after* the test period in some folds — a
direct leakage violation. Rolling-origin backtesting instead fixes three
origins (`d_1857`, `d_1885`, `d_1913`), each 28 days apart, and for each
one trains only on the trailing 730 days up to that origin and tests only
on the 28 days after it. `tests/test_splits.py` checks this against real
calendar-joined dates pulled from the warehouse — not just index
arithmetic — confirming `max(train date) < min(test date)` for every fold.
Three folds instead of one also surfaces whether performance is stable
across different periods, rather than telling you about one possibly
unusual 28-day window.

**Q8: Your bias flips sign fold to fold — what's going on, and why didn't
you fix it?**

Mean bias across the three folds is +0.002, which reads as essentially
neutral. But that average hides the actual pattern: fold 1 has bias
-0.025 (slight under-forecast), and folds 2 and 3 have +0.007 and +0.023
(slight over-forecast). I noticed this while writing up the M4 results and
made a deliberate choice not to paper over it by reporting only the mean —
averaging two opposite-signed numbers to near-zero and stopping there
would be technically true and substantively misleading. I don't have an
explanation for it yet; investigating it was scoped into M7's error
segmentation (by department, volume decile, intermittency, SNAP/event
weeks), which was cut when the project froze at M4. It's flagged
explicitly in the README's "what I'd do next" rather than either fixed
with an unverified guess or hidden behind the aggregate number.

**Q9: Walk me through the 3-store, 2-department scoping decision.**

Scoped to CA_3, TX_1, WI_2 (one store per state, needed for a SNAP
robustness check if that work is ever picked back up) and FOODS_3,
HOUSEHOLD_1 (the two highest-volume departments), giving ~4,065 series out
of the full 30,490. The reasoning: the SQL feature engineering, the
leakage-safety design, and the LightGBM architecture are identical at 4k
series and 30k series — `stores` and `depts` are config values in
`config.yaml`, not hardcoded anywhere in the pipeline. What changes with
series count is backtest iteration speed, and I chose to spend the saved
time on a 3-fold rolling-origin backtest, a full 8-config hyperparameter
grid with every config's numbers logged, and reporting against the
strongest baseline rather than the weakest one — all of which matter more
for defensibility than raw row count.

**Q10: Does `anchor_stride_days=14` bias the model?**

It's a legitimate question, and I don't have a controlled experiment that
rules it out — I'd flag that honestly rather than assert it's fine.
What I can say: the subsampling is applied to which *anchor days* seed
training examples, not to which series or which target days are ever
predicted or scored — every series and every day-of-week/season still
appears in training, just via fewer anchor points per fold (~5.7M rows/
fold instead of ~80M at daily density). The parameter exists because this
project ran on a 16GB machine and the full-density matrix caused an
out-of-memory failure — it's a config value, not a hardcoded shortcut, and
someone with more RAM could set it back to 1. But I have not run the
daily-density (stride=1) version to confirm the backtest numbers are
insensitive to the stride — that comparison wasn't done, and until it is,
"the results are unaffected by stride" is an assumption, not a verified
claim.

**Q11: Tell me about the memory bug — what went wrong and how did you
diagnose it?**

The first full run of the 8-config hyperparameter tuning grid climbed to
54GB of memory on a 16GB machine before I killed it — that's not "slow,"
that's a real bug. I traced it to two compounding issues in the tuning
loop: `lgb.Dataset(..., free_raw_data=False)` explicitly kept LightGBM's
internal copy of the raw pandas data alive after each Dataset was built
(normally freed once LightGBM has its own binary structure), and the loop
called `.copy()` on a ~10M-row training split fresh on every one of 8
iterations with no `del` or `gc.collect()` in between — so multiple full
copies plus 8 retained raw-data references accumulated across the loop
instead of being released. I confirmed the diagnosis empirically: I
watched actual process memory via `tasklist` on a single isolated config
before and after switching to `free_raw_data=True` (the library default;
nothing in this codebase reuses a Dataset after training) and adding
explicit cleanup — memory stayed bounded on the retest — and only reran
the full 8-config grid after verifying the fix on one config, rather than
re-running the expensive job blind and hoping it would behave differently.

**Q12: How would you explain this forecast to a category manager who
doesn't care about RMSSE?**

I wouldn't lead with RMSSE at all. I'd say: "For FOODS_3 and HOUSEHOLD_1
items at these three stores, this model's 28-day-ahead forecast is about
5% more accurate, dollar-weighted by what actually sells, than just
averaging the last 4 weeks — and unlike a rolling average, it can react to
a known upcoming holiday, a SNAP benefit day, or a planned price change,
because those are things it was explicitly given as inputs." I'd also be
upfront about the honest caveat, not hide it: on this particular slice the
margin over a simple 28-day average is modest (5.2%), so the case for
using a trained model over a spreadsheet-computable baseline rests more on
its ability to react to known future events than on raw average accuracy —
and I don't yet have the evidence to say how much of that reactivity
translates into value on the days that matter most (this is exactly what
M7's segmentation and M5's cost-of-error framing were meant to establish,
and neither was built).

**Q13: How would you productionize this?**

As built, this is a batch-retrain-and-backtest pipeline, not a service —
`warehouse.py build` regenerates the DuckDB tables from scratch, and
`lgbm_backtest.py`'s functions train fresh models per fold rather than
loading a persisted model for inference. To productionize: (1) separate
the training job (rebuild `feature_matrix`, retrain, validate against the
3-fold backtest as a regression gate before promoting a new model) from an
inference path that loads `models/lgbm_tweedie_final.txt` and scores new
anchor dates without rebuilding the whole warehouse; (2) add monitoring
for the kind of fold-to-fold bias drift flagged in Q8, since a sign flip
that's invisible in a backtest average could be very visible in production
if it correlates with a season or a promotional calendar; (3) decide a
retraining cadence tied to how often `sell_prices`/`calendar` actually
change meaningfully, not a fixed schedule; (4) the memory constraints
discovered on this dev machine (Q11) would need real profiling on
production hardware — the `anchor_stride_days` workaround was a
development-time compromise, not a production architecture decision.

**Q14: What's the difference between what M0-M4 built and what M5-M9
would have added?**

M0-M4 is one complete, tested loop: raw CSVs → a DuckDB warehouse with
scope filtering and leakage-safe features → baselines → a validated
global LightGBM model, with every number backed by a real backtest run and
every design choice logged with its rejected alternative in
`docs/decisions.md`. M5-M9 were independent analysis extensions that
*consume* that pipeline's output rather than complete it: M5 would convert
forecast error into a dollar cost-of-error number (newsvendor framing);
M6 would measure whether promotions in this data causally lift sales
(DiD, event study, power analysis); M7 would explain the model with SHAP
and segment its errors; M8/M9 were write-up and interview-prep milestones.
None of those are gaps in the forecasting pipeline itself — they're
separate questions the same feature/model foundation could answer if
built. `src/shelfsense/cost.py` and `plots.py` exist as unintegrated
scaffolding from before the freeze decision; they're explicitly called
out in the README as not part of the validated, tested scope.

**Q15: If I told you the model has to ship next week with no changes,
what would you want to check first, and what would you tell the business
about the risk?**

First, I'd want the M7 segmentation I don't have — specifically whether
the fold-to-fold bias flip (Q8) concentrates on a specific store, dept, or
volume band, because a bias that's invisible in the 3-fold average could
be a real problem for a specific high-revenue segment. Second, I'd want to
confirm the `anchor_stride_days=14` subsampling (Q10) doesn't materially
change the backtest numbers versus daily-density training, since that's
an assumption I haven't verified. To the business, I'd say: this model is
validated to beat a strong, honest baseline by 5.2% dollar-weighted,
across three independent 28-day periods, with a tested leakage-safety
guarantee — that's real and defensible. What I can't yet tell them is
*where* that 5.2% comes from (which segments, which conditions) or
whether there's a systematic direction to the remaining error that would
matter for inventory decisions specifically. I'd recommend shipping with
explicit monitoring on the bias-by-segment question rather than either
blocking the launch on unfinished analysis or shipping silently as if
that question were already answered.
