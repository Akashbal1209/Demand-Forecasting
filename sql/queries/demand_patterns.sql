-- How the 100 selected products split across demand patterns (smooth, erratic,
-- intermittent, lumpy), and how often each group has a week with no sales.

WITH product_weeks AS (
    SELECT stock_code,
           SUM(units)      AS units,
           SUM(units = 0)  AS zero_weeks,
           COUNT(*)        AS weeks
    FROM weekly_demand
    GROUP BY stock_code
)
SELECT sp.pattern,
       COUNT(*)                                          AS products,
       ROUND(AVG(sp.adi), 2)                             AS avg_adi,
       ROUND(AVG(sp.cv2), 2)                             AS avg_cv2,
       SUM(pw.units)                                     AS units,
       ROUND(100.0 * SUM(pw.zero_weeks) / SUM(pw.weeks), 1) AS pct_zero_weeks
FROM selected_products AS sp
JOIN product_weeks AS pw ON pw.stock_code = sp.stock_code
GROUP BY sp.pattern
ORDER BY products DESC;
