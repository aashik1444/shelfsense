-- ============================================================================
-- LEAKAGE RULE
--
-- You are forecasting 28 days ahead. On the forecast date t, you know sales
-- up to t-1 only. Therefore every lag and rolling feature must be shifted by
-- at least h=28. Price, calendar, SNAP and event features are *plans*, known
-- in advance, and may be used at time t directly.
--
-- The shift is implemented via the window FRAME clause, not by lagging then
-- rolling: AVG(units) OVER (PARTITION BY id ORDER BY date
--                            ROWS BETWEEN 55 PRECEDING AND 28 PRECEDING)
-- The frame's upper bound (28 PRECEDING) *is* the shift -- the most recent
-- row the frame can see is 28 days before the current row, so nothing
-- inside the frame was observed after t-28. This is safer than computing a
-- rolling feature first and then LAG()-ing the whole column by 28, because
-- that two-step version is a single off-by-one mistake away from leaking
-- (e.g. rolling with a frame that includes CURRENT ROW, then lagging by 27
-- instead of 28) with no error thrown -- it just silently trains on data it
-- shouldn't see.
-- ============================================================================

CREATE OR REPLACE TABLE feature_matrix AS
WITH lag_and_roll AS (
    SELECT
        id, item_id, dept_id, cat_id, store_id, state_id,
        date, d, units,

        -- lag features, all shifted >= 28 by construction (LAG(col, 28) means
        -- "the value 28 rows before this one")
        LAG(units, 28) OVER w AS lag_28,
        LAG(units, 35) OVER w AS lag_35,
        LAG(units, 42) OVER w AS lag_42,
        LAG(units, 56) OVER w AS lag_56,
        LAG(units, 364) OVER w AS lag_364,

        -- rolling means: window ends at t-28, the frame clause is the shift
        AVG(units) OVER (
            PARTITION BY id ORDER BY date ROWS BETWEEN 34 PRECEDING AND 28 PRECEDING
        ) AS roll_mean_7,
        AVG(units) OVER (
            PARTITION BY id ORDER BY date ROWS BETWEEN 55 PRECEDING AND 28 PRECEDING
        ) AS roll_mean_28,
        AVG(units) OVER (
            PARTITION BY id ORDER BY date ROWS BETWEEN 83 PRECEDING AND 28 PRECEDING
        ) AS roll_mean_56,
        AVG(units) OVER (
            PARTITION BY id ORDER BY date ROWS BETWEEN 207 PRECEDING AND 28 PRECEDING
        ) AS roll_mean_180,

        -- rolling std
        STDDEV(units) OVER (
            PARTITION BY id ORDER BY date ROWS BETWEEN 55 PRECEDING AND 28 PRECEDING
        ) AS roll_std_28,
        STDDEV(units) OVER (
            PARTITION BY id ORDER BY date ROWS BETWEEN 83 PRECEDING AND 28 PRECEDING
        ) AS roll_std_56,

        -- rolling max
        MAX(units) OVER (
            PARTITION BY id ORDER BY date ROWS BETWEEN 55 PRECEDING AND 28 PRECEDING
        ) AS roll_max_28,

        -- intermittency: share of zero-demand days in the trailing 28-day
        -- window ending at t-28
        AVG(CASE WHEN units = 0 THEN 1.0 ELSE 0.0 END) OVER (
            PARTITION BY id ORDER BY date ROWS BETWEEN 55 PRECEDING AND 28 PRECEDING
        ) AS roll_zero_share_28

    FROM sales_long
    WINDOW w AS (PARTITION BY id ORDER BY date)
),
-- current zero-run length as of t-28: the number of consecutive zero-unit
-- days immediately preceding (and including) t-28. Computed with the
-- classic "row_number minus row_number-of-last-nonzero-run" gap-and-island
-- trick, then shifted so the value used for row t is the run length as it
-- stood at t-28, not at t.
zero_run_calc AS (
    SELECT
        id,
        date,
        units,
        ROW_NUMBER() OVER (PARTITION BY id ORDER BY date)
            - ROW_NUMBER() OVER (
                PARTITION BY id, (CASE WHEN units = 0 THEN 0 ELSE 1 END)
                ORDER BY date
              ) AS zero_run_group
    FROM sales_long
),
zero_run_length AS (
    SELECT
        id,
        date,
        units,
        CASE
            WHEN units = 0
            THEN COUNT(*) OVER (PARTITION BY id, zero_run_group ORDER BY date
                                 ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
            ELSE 0
        END AS zero_run_length_asof_t
    FROM zero_run_calc
),
zero_run_shifted AS (
    SELECT
        id,
        date,
        LAG(zero_run_length_asof_t, 28) OVER (PARTITION BY id ORDER BY date) AS zero_run_length
    FROM zero_run_length
),
snap_by_state AS (
    SELECT
        id,
        date,
        CASE state_id
            WHEN 'CA' THEN snap_CA
            WHEN 'TX' THEN snap_TX
            WHEN 'WI' THEN snap_WI
        END AS snap
    FROM sales_long
)
SELECT
    lr.id, lr.item_id, lr.dept_id, lr.cat_id, lr.store_id, lr.state_id,
    lr.date, lr.units,

    -- days_since_first_sale: series meta, known at forecast time (it's a
    -- fact about the series' own age, not about future sales)
    DATE_DIFF('day', MIN(lr.date) OVER (PARTITION BY lr.id), lr.date) AS days_since_first_sale,

    lr.lag_28, lr.lag_35, lr.lag_42, lr.lag_56, lr.lag_364,
    lr.roll_mean_7, lr.roll_mean_28, lr.roll_mean_56, lr.roll_mean_180,
    lr.roll_std_28, lr.roll_std_56,
    lr.roll_max_28,
    lr.roll_zero_share_28,
    zrs.zero_run_length,

    pf.sell_price,
    pf.price_vs_item_median_8w,
    pf.price_vs_dept_median_week,
    pf.price_changed_flag,
    pf.price_momentum_4w,

    cf.dow, cf.day_of_month, cf.week_of_year, cf.month, cf.year, cf.is_weekend,
    sbs.snap,
    cf.event_type_1, cf.days_to_next_event, cf.days_since_last_event,
    cf.is_christmas

FROM lag_and_roll lr
JOIN zero_run_shifted zrs ON lr.id = zrs.id AND lr.date = zrs.date
JOIN price_features pf ON lr.id = pf.id AND lr.date = pf.date
JOIN calendar_features cf ON lr.date = cf.date
JOIN snap_by_state sbs ON lr.id = sbs.id AND lr.date = sbs.date
;
