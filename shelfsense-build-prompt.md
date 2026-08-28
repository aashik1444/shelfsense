# shelfsense — Claude Code Build Prompt

**How to use this file**

1. Create an empty folder `shelfsense/`, `cd` into it, run `claude`.
2. Copy `shelfsense-milestones.md` into the folder as `MILESTONES.md`.
3. Paste **Section A** below as your first message. That kicks off M0–M1.
4. After each milestone completes and you've reviewed it, paste the matching line from **Section B**.
5. Use **Section C** when something goes sideways.

Do not paste the whole thing at once. One milestone per turn is what keeps the output reviewable and stops it inventing scope.

---

## SECTION A — The kickoff prompt

```
You are helping me build a portfolio data science project called shelfsense. I am an
EEE final-year student at NIT Calicut interviewing for an Associate Data Scientist role
at Tredence, which runs two technical interview rounds. I will personally have to defend
every line of this code in those interviews.

Read MILESTONES.md in this directory in full before writing anything. It is the
specification. Follow it exactly.

## Non-negotiable constraints

1. SCOPE DISCIPLINE. Build only what MILESTONES.md specifies. If you think something
   else would improve the project, say so in one sentence and wait for me to decide.
   Do not add it. The most common way this project fails is scope creep.

2. NO NEW DEPENDENCIES beyond: duckdb, pandas, numpy, lightgbm, scikit-learn,
   statsmodels, shap, matplotlib, pyyaml, pytest. If you believe one is genuinely
   required, ask first with a one-line justification.

3. SQL IS A FIRST-CLASS DELIVERABLE. All feature engineering lives in .sql files in
   sql/, not in pandas. Use CTEs, window functions with explicit frame clauses, and
   QUALIFY where it helps. An interviewer will open these files. Write them to be read:
   comment each CTE with what it produces and why.

4. LEAKAGE IS THE FAILURE MODE THAT MATTERS. The forecast horizon is 28 days. Every
   lag and rolling feature must be computed from data at or before t-28. Implement the
   shift via the window frame (ROWS BETWEEN 55 PRECEDING AND 28 PRECEDING), not by
   lagging then rolling. Write tests/test_leakage.py before you write the model, and
   make it a real test that reconstructs feature values from raw data.

5. CONFIG-DRIVEN. Every parameter — stores, departments, horizon, cost rates, promo
   threshold, seeds — lives in config.yaml and is read through src/shelfsense/config.py.
   Zero magic numbers anywhere else. I need to change the store list in one place.

6. REPRODUCIBLE. Fixed seeds. A fresh clone plus three commands must reproduce the
   headline numbers. Verify this yourself at M8; don't assume it.

7. HONEST NUMBERS. Never write a placeholder result, a rounded-up figure, or a number
   you did not compute. If a model underperforms a baseline on some segment, report
   that. I would rather defend a real 12% improvement than get caught on a fake 30%.

8. TELL ME WHEN I'M WRONG. If a design choice in MILESTONES.md is technically wrong or
   won't work in this dataset, say so directly and propose the fix. Do not silently
   work around it.

## Working style

- Work one milestone at a time. Stop at the end of each milestone, print the
  "Definition of done" checklist from MILESTONES.md with each item marked pass/fail,
  and wait for me before continuing.
- Explain design decisions as you make them, briefly, in the chat — not in long code
  comments. Append each real decision (with the alternative you rejected and why) to
  docs/decisions.md as a dated entry. That file is what I revise from before the
  interview.
- Prefer boring, readable code over clever code. I have to explain it out loud.
- Type hints on public functions. Docstrings only where the function's purpose isn't
  obvious from its name.
- Commit at the end of each milestone with a message naming the milestone.

## Your first task: M0 and M1

Do M0 (scaffold, config, data verification) and M1 (DuckDB warehouse, wide-to-long,
scope filter, price and calendar joins, first-sale trimming) exactly as MILESTONES.md
specifies.

The M5 CSVs are already in data/raw/: calendar.csv, sales_train_evaluation.csv,
sell_prices.csv, sample_submission.csv.

Before you write code, tell me in under 150 words:
- how you'll do the wide-to-long unpivot in DuckDB
- how you'll handle the leading zero-runs before an item's first sale
- anything in MILESTONES.md M0/M1 you think is wrong

Then build it.
```

---

## SECTION B — Per-milestone continuation prompts

Paste one at a time, after reviewing the previous milestone's output.

### → M2 (SQL feature layer)
```
M1 approved. Proceed to M2: the SQL feature layer.

Before writing sql/05_feature_matrix.sql, produce the known-at-forecast-time table for
every feature you plan to build — feature name, source, shift applied, known in advance
yes/no. I want to review that table before you write the SQL, because getting this wrong
invalidates everything downstream.

Then build the SQL, then write tests/test_leakage.py as MILESTONES.md specifies — it
must independently reconstruct feature values from sales_long using only date <= t-28
and assert equality on a random sample of 200 rows. Show me the test passing.
```

### → M3 (metrics, baselines, backtest)
```
M2 approved. Proceed to M3: metrics, baselines, and the rolling-origin backtest harness.

Implement RMSSE carefully — the scaling denominator is the mean squared one-step naive
error computed on that fold's TRAINING window only. Show me a unit test proving that a
naive forecast scores approximately 1.0.

Run all four baselines across all three folds and give me the results table. Tell me
which baseline is hardest to beat and why you think that is.
```

### → M4 (LightGBM)
```
M3 approved. Proceed to M4: the global LightGBM model.

Direct multi-horizon with days_ahead as a feature, not recursive. Tweedie objective as
the primary, with a logged comparison against poisson and l2 on fold 1.

Cap tuning at the 8 hand-specified configurations in MILESTONES.md. Log all 8. I want
to be able to say in an interview that the spread between the best and median config
was small relative to the gap over the baseline — so give me those exact numbers.

Report weighted RMSSE, MAE and bias per fold, against every baseline from M3.
```

### → M5 (cost of error)
```
M4 approved. Proceed to M5: quantile forecasts and the cost-of-error simulation.

This is the most important milestone in the project. Derive the critical ratio from
config, train the quantile model at that alpha, and simulate all four policies (P0-P3)
over the test folds.

Give me: the cost table across three margin assumptions, the service-level-vs-cost
chart with the newsvendor optimum marked, and one sentence stating the headline finding
with a real dollar figure and the extrapolation assumption made explicit.
```

### → M6 (uplift)
```
M5 approved. Proceed to M6: promotional uplift.

Build the promo panel in SQL (sql/06_promo_panel.sql), then run in this order:
1. DiD with store-item and week fixed effects, SEs clustered at store-item
2. Event study, weeks -4 to +4, week -1 omitted, plotted with CIs
3. Placebo test with promo dates shifted 8 weeks earlier
4. Power analysis: MDE at 80% power for my actual N, plus the inverted table
   (N required for a 5% and a 2% lift)

Then write the limitations paragraph and the "how I'd design this as a randomised
experiment" section for results.md.

Be blunt with me about the parallel-trends result. If the pre-period coefficients trend
upward, say so plainly and explain what it means for the estimate — I would rather walk
into the interview knowing my design's weakness than be told it there.
```

### → M7 (interpretability)
```
M6 approved. Proceed to M7: SHAP and error diagnostics.

Sample for SHAP, don't run the full test set. Global beeswarm and importance plots,
then two local explanations written out as prose narratives — one large over-forecast,
one large under-forecast.

Then the error segmentation table across department, store, volume decile,
intermittency, SNAP day, and event week. Finish with a short honest paragraph on where
the model is weakest.
```

### → M8 (write-up)
```
M7 approved. Proceed to M8: the write-up.

Three artifacts: reports/results.md (technical, ~1500 words, 10 sections per
MILESTONES.md), reports/one_pager.md (exec memo, ~250 words, recommendation first),
and README.md with headline results above the fold.

Then actually verify reproducibility: from a clean state, run the three setup commands
and confirm the numbers in the README match what the pipeline produces. Report any
mismatch rather than editing the README to fit.

Every figure needs labelled axes and must be legible as a thumbnail.
```

### → M9 (interview drill)
```
M8 approved. Final milestone, M9: docs/interview_drill.md.

Write full-sentence answers to all 20 questions in MILESTONES.md M9, using the actual
numbers this project produced — not generic answers. Then add the 60-second and
5-minute walkthrough scripts and the three headline numbers.

Then switch roles: interview me. Ask me five of those questions one at a time, wait for
my answer, and grade it honestly against what the code actually does. Point out where
I'm hand-waving.
```

---

## SECTION C — Recovery and pressure-test prompts

### Leakage audit
```
Stop and audit the entire pipeline for leakage. For every feature in feature_matrix,
trace back to the SQL that produces it and confirm no value at time t uses information
from after t-28, except for features I explicitly classified as known-in-advance
(price, calendar, SNAP, events). List anything suspicious. Do not tell me it's fine
without showing me the check.
```

### Results look too good
```
My weighted RMSSE improvement over baseline is [X]%. That seems high for M5. Before I
put this on a resume, find the reason it might be wrong. Check in this order: leakage
in rolling features, the RMSSE denominator using the wrong window, test rows appearing
in training, and the Christmas zero-sales days inflating the baseline error.
```

### Falling behind schedule
```
I've lost a day. Re-plan the remaining milestones to fit [N] hours, cutting the least
valuable work. Constraint: M5 (cost of error), M6 (uplift) and M9 (interview drill)
cannot be cut — they are the entire differentiation of this project. Tell me exactly
what you're dropping and what it costs me.
```

### Adversarial review before the interview
```
Read the whole repo and results.md. Act as a skeptical senior data scientist
interviewing me. Give me the ten hardest questions you'd ask, ranked by how badly they'd
expose a weakness — then tell me the honest answer to each based on what the code
actually does, including the ones where the honest answer is "that's a real limitation".
```

### Resume bullets
```
Read results.md and produce three resume bullets in the format in MILESTONES.md, using
only numbers this project actually produced. No rounding up, no vague verbs. Each bullet
should name the technique, the outcome and a figure. Then flag any bullet an interviewer
could ask me a question about that I couldn't answer from this codebase.
```

---

## A note on how to run this

Review the code at every milestone gate before pasting the next prompt. Two technical
rounds will find the parts you didn't read. If a milestone produces something you don't
understand, ask Claude Code to explain it before you approve — that conversation is
prep, not a detour.
