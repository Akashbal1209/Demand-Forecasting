-- Number of orders (distinct invoices) per selected product per week.
-- Same week rules as weekly_demand.sql (Monday weeks, partial first and last
-- weeks dropped, 0 when a product has no orders), so the two tables line up.
-- Used for the order-based model features in notebook 07.

WITH sale_weeks AS (
    SELECT stock_code,
           date(invoice_date, 'weekday 0', '-6 days') AS week,
           invoice
    FROM sales
),
week_range AS (
    SELECT date(MIN(week), '+7 days') AS first_week,
           date(MAX(week), '-7 days') AS last_week
    FROM sale_weeks
),
weeks (week) AS (
    SELECT first_week FROM week_range
    UNION ALL
    SELECT date(week, '+7 days')
    FROM weeks
    WHERE week < (SELECT last_week FROM week_range)
),
product_weeks AS (
    SELECT stock_code, week, COUNT(DISTINCT invoice) AS orders
    FROM sale_weeks
    WHERE stock_code IN (SELECT stock_code FROM selected_products)
    GROUP BY stock_code, week
)
SELECT sp.stock_code,
       w.week,
       COALESCE(pw.orders, 0) AS orders
FROM selected_products AS sp
CROSS JOIN weeks AS w
LEFT JOIN product_weeks AS pw
       ON pw.stock_code = sp.stock_code
      AND pw.week = w.week
ORDER BY sp.stock_code, w.week;
