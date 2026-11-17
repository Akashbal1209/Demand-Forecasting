-- Units, revenue, orders and customers per month,
-- plus revenue change against the same month one year earlier.

WITH monthly AS (
    SELECT strftime('%Y-%m', invoice_date) AS month,
           SUM(quantity)                   AS units,
           ROUND(SUM(revenue), 2)          AS revenue,
           COUNT(DISTINCT invoice)         AS invoices,
           COUNT(DISTINCT customer_id)     AS customers
    FROM sales
    GROUP BY month
)
SELECT month,
       units,
       revenue,
       invoices,
       customers,
       ROUND(revenue / invoices, 2) AS avg_invoice_value,
       ROUND(100.0 * (revenue - LAG(revenue, 12) OVER (ORDER BY month))
                   / LAG(revenue, 12) OVER (ORDER BY month), 1) AS revenue_yoy_pct
FROM monthly
ORDER BY month;
