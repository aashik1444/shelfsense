-- Load the four M5 CSVs as-is into raw DuckDB tables. No filtering, no
-- transformation here — this file's only job is "CSV on disk -> table in
-- DuckDB" so every later SQL file has a stable, typed starting point.

CREATE OR REPLACE TABLE raw_calendar AS
SELECT *
FROM read_csv_auto('data/raw/calendar.csv');

CREATE OR REPLACE TABLE raw_sales AS
SELECT *
FROM read_csv_auto('data/raw/sales_train_evaluation.csv');

CREATE OR REPLACE TABLE raw_prices AS
SELECT *
FROM read_csv_auto('data/raw/sell_prices.csv');
