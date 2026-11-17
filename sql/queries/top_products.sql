-- Best-selling products: rank by units sold and by revenue.
-- Parameter :top_n = how many products to return.

WITH product_totals AS (
    SELECT s.stock_code,
           p.name,
           SUM(s.quantity)           AS units,
           ROUND(SUM(s.revenue), 2)  AS revenue,
           COUNT(DISTINCT s.invoice) AS invoices
    FROM sales AS s
    JOIN products AS p ON p.stock_code = s.stock_code
    GROUP BY s.stock_code, p.name
)
SELECT RANK() OVER (ORDER BY units DESC)   AS units_rank,
       RANK() OVER (ORDER BY revenue DESC) AS revenue_rank,
       stock_code,
       name,
       units,
       revenue,
       invoices
FROM product_totals
ORDER BY units_rank
LIMIT :top_n;
