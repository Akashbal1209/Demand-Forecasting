-- Weekly demand for the 100 selected products, built from the clean sales lines.
-- Same rules as notebook 02:
--   * a week starts on Monday
--   * the first and last weeks of the data are partial, so they are dropped
--   * a week with no sales gets 0 units
-- Result: one row per product per week.

WITH sale_weeks AS (
    SELECT stock_code,
           date(invoice_date, 'weekday 0', '-6 days') AS week,   -- Monday of the sale's week
           quantity
    FROM sales
),
week_range AS (
    SELECT date(MIN(week), '+7 days') AS first_week,              -- skip the partial first week
           date(MAX(week), '-7 days') AS last_week                -- skip the partial last week
    FROM sale_weeks
),
weeks (week) AS (                                                 -- every Monday in the range
    SELECT first_week FROM week_range
    UNION ALL
    SELECT date(week, '+7 days')
    FROM weeks
    WHERE week < (SELECT last_week FROM week_range)
),
product_weeks AS (
    SELECT stock_code, week, SUM(quantity) AS units
    FROM sale_weeks
    WHERE stock_code IN (SELECT stock_code FROM selected_products)
    GROUP BY stock_code, week
)
SELECT sp.stock_code,
       w.week,
       COALESCE(pw.units, 0) AS units
FROM selected_products AS sp
CROSS JOIN weeks AS w
LEFT JOIN product_weeks AS pw
       ON pw.stock_code = sp.stock_code
      AND pw.week = w.week
ORDER BY sp.stock_code, w.week;
