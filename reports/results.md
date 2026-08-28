# shelfsense — Results

## Baselines (M3)

3-fold rolling-origin backtest, dollar-weighted RMSSE, on the 3-store x
2-dept scope (4,065 series). Train window: trailing 730 days from each
origin. Test window: 28 days immediately after the origin.

| origin | naive | seasonal_naive_7 | mean_28 | croston_lite |
|---|---|---|---|---|
| d_1857 (2016-02-28) | 1.086 | 1.099 | 0.840 | 0.904 |
| d_1885 (2016-03-27) | 1.062 | 1.087 | 0.835 | 0.906 |
| d_1913 (2016-04-24) | 1.049 | 1.067 | 0.810 | 0.875 |
| **mean (weighted RMSSE)** | **1.066** | **1.084** | **0.829** | **0.895** |
| mean MAE | 1.939 | 1.976 | 1.629 | 2.019 |
| mean bias | +0.012 | -0.005 | -0.035 | +0.857 |

`mean_28` (trailing 28-day mean) is the strongest baseline, as the spec
anticipated — for a stable, slow-moving product mix like FOODS_3 and
HOUSEHOLD_1, a simple trailing average already captures most of the
predictable signal. `croston_lite` is biased strongly positive (+0.857):
it forecasts the average size of a non-zero demand event every day,
without down-weighting for how often demand actually is zero, so it
systematically over-forecasts series with meaningful intermittency. Any
LightGBM model (M4) needs to beat `mean_28`'s weighted RMSSE of 0.829 to
be worth the added complexity.
