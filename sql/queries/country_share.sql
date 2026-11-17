-- Revenue by country and each country's share of total revenue.
-- Parameter :top_n = how many countries to return.

SELECT country,
       COUNT(DISTINCT customer_id) AS customers,
       COUNT(DISTINCT invoice)     AS invoices,
       ROUND(SUM(revenue), 2)      AS revenue,
       ROUND(100.0 * SUM(revenue) / SUM(SUM(revenue)) OVER (), 1) AS revenue_share_pct
FROM sales
GROUP BY country
ORDER BY revenue DESC
LIMIT :top_n;
