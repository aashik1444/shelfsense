-- Price features. Prices are a *plan*, not an outcome -- a retailer sets
-- next week's price in advance, so sell_price and everything derived from
-- it at time t is known at forecast time and needs no leakage shift.

CREATE OR REPLACE TABLE price_features AS
SELECT
    id,
    date,
    sell_price,
    -- trailing 8-week median: 8 weeks = 56 days, current day included since
    -- the current price is already known (it's a plan, not an outcome)
    sell_price / NULLIF(
        MEDIAN(sell_price) OVER (
            PARTITION BY id
            ORDER BY date
            ROWS BETWEEN 55 PRECEDING AND CURRENT ROW
        ),
        0
    ) AS price_vs_item_median_8w,
    -- median price across the same dept-store on the same calendar week --
    -- "am I discounted relative to my peers this week"
    sell_price / NULLIF(
        MEDIAN(sell_price) OVER (
            PARTITION BY dept_id, store_id, wm_yr_wk
        ),
        0
    ) AS price_vs_dept_median_week,
    CASE
        WHEN sell_price != LAG(sell_price) OVER (PARTITION BY id ORDER BY date)
        THEN 1 ELSE 0
    END AS price_changed_flag,
    -- momentum: current price vs price 4 weeks (28 days) ago -- a plan-vs-plan
    -- comparison, not a lag on the outcome variable, so no h=28 shift needed
    sell_price / NULLIF(
        LAG(sell_price, 28) OVER (PARTITION BY id ORDER BY date),
        0
    ) AS price_momentum_4w
FROM sales_long
;
