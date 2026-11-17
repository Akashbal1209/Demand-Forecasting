-- Customers with many orders, and how much of total revenue they bring.
-- Guest checkouts (no customer_id) are left out.
-- Parameters: :min_invoices = minimum number of orders, :top_n = rows to return.

WITH customer_totals AS (
    SELECT customer_id,
           country,
           COUNT(DISTINCT invoice)                 AS invoices,
           SUM(quantity)                           AS units,
           ROUND(SUM(revenue), 2)                  AS revenue,
           SUM(is_large_line)                      AS large_lines,
           MIN(date(invoice_date))                 AS first_order,
           MAX(date(invoice_date))                 AS last_order
    FROM sales
    WHERE customer_id IS NOT NULL
    GROUP BY customer_id, country
    HAVING COUNT(DISTINCT invoice) >= :min_invoices
)
SELECT customer_id,
       country,
       invoices,
       units,
       revenue,
       ROUND(100.0 * revenue / (SELECT SUM(revenue) FROM sales), 2) AS revenue_share_pct,
       large_lines,
       first_order,
       last_order
FROM customer_totals
ORDER BY revenue DESC
LIMIT :top_n;
