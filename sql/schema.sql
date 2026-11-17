-- Schema for data/retail.db: the full analysis database.
-- Built by scripts/build_db.py from the files in data/processed/.
-- Dates are stored as ISO text ('YYYY-MM-DD HH:MM:SS'), which SQLite date functions understand.

PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS weekly_demand;
DROP TABLE IF EXISTS selected_products;
DROP TABLE IF EXISTS sales;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS cleaning_log;

-- one row per product: its most common description
CREATE TABLE products (
    stock_code  TEXT PRIMARY KEY,
    name        TEXT NOT NULL
);

-- one row per clean invoice line (output of notebook 01)
CREATE TABLE sales (
    sale_id        INTEGER PRIMARY KEY,
    invoice        TEXT    NOT NULL,
    stock_code     TEXT    NOT NULL REFERENCES products (stock_code),
    description    TEXT    NOT NULL,
    quantity       INTEGER NOT NULL CHECK (quantity > 0),
    invoice_date   TEXT    NOT NULL,
    price          REAL    NOT NULL CHECK (price > 0),
    customer_id    INTEGER,                      -- NULL = guest checkout
    country        TEXT    NOT NULL,
    revenue        REAL    NOT NULL,
    is_large_line  INTEGER NOT NULL CHECK (is_large_line IN (0, 1))
);

CREATE INDEX idx_sales_stock_code ON sales (stock_code);
CREATE INDEX idx_sales_invoice_date ON sales (invoice_date);

-- the 100 products chosen for forecasting (notebook 02), in sales-rank order
CREATE TABLE selected_products (
    stock_code    TEXT PRIMARY KEY REFERENCES products (stock_code),
    sales_rank    INTEGER NOT NULL UNIQUE,
    capped_total  REAL    NOT NULL,
    adi           REAL    NOT NULL,              -- average weeks between sales
    cv2           REAL    NOT NULL,              -- squared coefficient of variation
    pattern       TEXT    NOT NULL
);

-- units sold per product per week (week = Monday), zero-filled
CREATE TABLE weekly_demand (
    stock_code  TEXT    NOT NULL REFERENCES selected_products (stock_code),
    week        TEXT    NOT NULL,
    units       INTEGER NOT NULL,
    PRIMARY KEY (stock_code, week)
);

-- how many rows each cleaning rule removed (notebook 01)
CREATE TABLE cleaning_log (
    step             TEXT PRIMARY KEY,
    rows_before      INTEGER NOT NULL,
    rows_after       INTEGER NOT NULL,
    rows_removed     INTEGER NOT NULL,
    pct_of_raw_left  REAL    NOT NULL
);
