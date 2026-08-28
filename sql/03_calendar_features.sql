-- Calendar features. All of these are known in advance (they describe the
-- forecast date itself, not past sales), so none of them need a leakage
-- shift -- they can be computed directly on date t.

CREATE OR REPLACE TABLE calendar_features AS
WITH events AS (
    -- one row per calendar day that has *any* event (name_1 or name_2),
    -- used below to compute days_to_next_event / days_since_last_event
    SELECT DISTINCT date
    FROM raw_calendar
    WHERE event_name_1 IS NOT NULL OR event_name_2 IS NOT NULL
),
calendar_with_event_dist AS (
    SELECT
        c.date,
        c.d,
        c.wm_yr_wk,
        c.event_type_1,
        -- nearest event on/after this date, in days
        (
            SELECT MIN(e.date - c.date)
            FROM events e
            WHERE e.date >= c.date
        ) AS days_to_next_event,
        -- nearest event on/before this date, in days
        (
            SELECT MIN(c.date - e.date)
            FROM events e
            WHERE e.date <= c.date
        ) AS days_since_last_event
    FROM raw_calendar c
)
SELECT
    date,
    d,
    wm_yr_wk,
    DAYOFWEEK(date) AS dow,
    DAY(date) AS day_of_month,
    WEEKOFYEAR(date) AS week_of_year,
    MONTH(date) AS month,
    YEAR(date) AS year,
    CASE WHEN DAYOFWEEK(date) IN (0, 6) THEN 1 ELSE 0 END AS is_weekend,
    event_type_1,
    days_to_next_event,
    days_since_last_event,
    -- Walmart is closed Dec 25 every year; sales are structurally zero,
    -- not a demand signal. Flag it so the model doesn't learn "December"
    -- as a demand-collapse period.
    CASE WHEN MONTH(date) = 12 AND DAY(date) = 25 THEN 1 ELSE 0 END AS is_christmas
FROM calendar_with_event_dist
;
