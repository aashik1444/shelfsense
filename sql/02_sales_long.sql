-- Wide-to-long unpivot of raw_sales, scoped to the configured stores/depts,
-- joined to calendar (date, week key) and prices (weekly, per store-item),
-- with the pre-launch leading zero-run trimmed off each series.
--
-- Scope filter happens in the same statement as the unpivot so we never
-- materialize the full 30,490-series long table, only the ~4,000-series
-- slice this project actually models.

CREATE OR REPLACE TABLE sales_scoped AS
UNPIVOT raw_sales
ON COLUMNS(* EXCLUDE (id, item_id, dept_id, cat_id, store_id, state_id))
INTO
    NAME d
    VALUE units
;

-- filter to scope: applied after unpivot since UNPIVOT can't carry a WHERE
-- clause directly; DuckDB pushes the predicate down into the scan so this
-- is not a full-table-then-filter cost in practice.
-- {stores} / {depts} are substituted by warehouse.py from config.yaml
-- before this file is run (config values, not user input).
CREATE OR REPLACE TABLE sales_scoped AS
SELECT *
FROM sales_scoped
WHERE store_id IN ({stores})
  AND dept_id IN ({depts})
;

-- join calendar: d -> date, wm_yr_wk, and the SNAP/event columns needed by
-- later feature files. Every sales row must find exactly one calendar row.
CREATE OR REPLACE TABLE sales_with_calendar AS
SELECT
    s.id,
    s.item_id,
    s.dept_id,
    s.cat_id,
    s.store_id,
    s.state_id,
    s.d,
    s.units,
    c.date,
    c.wm_yr_wk,
    c.wday,
    c.month,
    c.year,
    c.event_name_1,
    c.event_type_1,
    c.event_name_2,
    c.event_type_2,
    c.snap_CA,
    c.snap_TX,
    c.snap_WI
FROM sales_scoped s
JOIN raw_calendar c
    ON s.d = c.d
;

-- join sell_prices on (store_id, item_id, wm_yr_wk). Prices are weekly;
-- this join fans a single weekly price out across every day in that week.
-- A store-item-week with no price row means the item was not yet stocked
-- at that store that week -- those rows are dropped by the first-sale trim
-- below, not by this join (an inner join here would silently drop days
-- *within* a stocked item's life if a price row were ever missing for a
-- reason other than "not yet stocked", so we keep the join and trim
-- explicitly instead of relying on join semantics to do it for us).
CREATE OR REPLACE TABLE sales_with_price AS
SELECT
    swc.*,
    p.sell_price
FROM sales_with_calendar swc
LEFT JOIN raw_prices p
    ON swc.store_id = p.store_id
   AND swc.item_id = p.item_id
   AND swc.wm_yr_wk = p.wm_yr_wk
;

-- first_sale_date per series: the first date with units > 0. Rows before
-- this date are "not yet stocked", not "zero demand" -- modelling them as
-- demand deflates the RMSSE scaling denominator and teaches the model a
-- fake near-zero baseline for the item's early life.
CREATE OR REPLACE TABLE first_sale AS
SELECT
    id,
    MIN(date) AS first_sale_date
FROM sales_with_price
WHERE units > 0
GROUP BY id
;

CREATE OR REPLACE TABLE sales_long AS
SELECT swp.*
FROM sales_with_price swp
JOIN first_sale fs
    ON swp.id = fs.id
WHERE swp.date >= fs.first_sale_date
;

DROP TABLE sales_scoped;
DROP TABLE sales_with_calendar;
DROP TABLE sales_with_price;
DROP TABLE first_sale;
