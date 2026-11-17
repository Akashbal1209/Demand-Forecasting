-- Units sold by the 100 selected products in each calendar month,
-- with one column per year so the two gift seasons can be compared side by side.

SELECT CAST(strftime('%m', s.invoice_date) AS INTEGER) AS month_no,
       SUM(CASE WHEN strftime('%Y', s.invoice_date) = '2010' THEN s.quantity ELSE 0 END) AS units_2010,
       SUM(CASE WHEN strftime('%Y', s.invoice_date) = '2011' THEN s.quantity ELSE 0 END) AS units_2011
FROM sales AS s
JOIN selected_products AS sp ON sp.stock_code = s.stock_code
GROUP BY month_no
ORDER BY month_no;
