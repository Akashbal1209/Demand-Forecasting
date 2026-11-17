-- Schema for data/app.db: the small database the FastAPI app reads.
-- Built by scripts/build_db.py. It holds only what the app needs, so it stays small
-- enough to ship with the code.

PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS weekly_demand;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS model_info;

-- the 100 products the model forecasts
CREATE TABLE products (
    stock_code  TEXT PRIMARY KEY,
    name        TEXT NOT NULL
);

-- weekly units the model was trained on (week = Monday)
CREATE TABLE weekly_demand (
    stock_code  TEXT    NOT NULL REFERENCES products (stock_code),
    week        TEXT    NOT NULL,
    units       INTEGER NOT NULL,
    PRIMARY KEY (stock_code, week)
);

-- facts about the saved model, e.g. trained_until and test metrics (JSON text)
CREATE TABLE model_info (
    key    TEXT PRIMARY KEY,
    value  TEXT NOT NULL
);
