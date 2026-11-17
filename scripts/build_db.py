"""
Build the two SQLite databases from the files made by the notebooks.

Run from the project root:
    python scripts/build_db.py

    data/retail.db : full analysis database (about 1 million clean sales lines).
                     Too big for GitHub, so it is not in git; this script rebuilds it.
    data/app.db    : small database read by the FastAPI app (100 products x 104 weeks).
                     It is kept in git so the deployed app has its data.
"""

import json
import sqlite3
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                    # so "src" can be imported when run as a script

from src.db import APP_DB, RETAIL_DB, SQL_DIR   # noqa: E402

PROCESSED_DIR = ROOT / "data" / "processed"
BUNDLE_PATH = ROOT / "models" / "forecast_bundle.pkl"


def create_db(path, schema_file):
    """Start a new, empty database file and create its tables from a schema file."""
    path.unlink(missing_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript((SQL_DIR / schema_file).read_text(encoding="utf-8"))
    return conn


def insert(conn, table, df):
    """Append a DataFrame to an existing table (column names must match the table)."""
    df.to_sql(table, conn, if_exists="append", index=False, chunksize=50_000)


def print_counts(conn, path):
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")]
    size_mb = path.stat().st_size / 1e6
    print(f"{path.relative_to(ROOT)} ({size_mb:.1f} MB)")
    for t in tables:
        n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  {t:<18} {n:>9,} rows")


# ---------------------------------------------------------------- retail.db
def build_retail_db():
    sales = pd.read_parquet(PROCESSED_DIR / "clean_sales.parquet")
    names = pd.read_csv(PROCESSED_DIR / "product_names.csv", dtype={"StockCode": "string"})
    selected = pd.read_csv(PROCESSED_DIR / "selected_products.csv", dtype={"StockCode": "string"})
    weekly = pd.read_parquet(PROCESSED_DIR / "weekly_demand.parquet")
    log = pd.read_csv(PROCESSED_DIR / "cleaning_log.csv")

    conn = create_db(RETAIL_DB, "schema.sql")

    insert(conn, "products", pd.DataFrame({
        "stock_code": names["StockCode"],
        "name": names["Description"],
    }))

    insert(conn, "sales", pd.DataFrame({
        "invoice": sales["Invoice"],
        "stock_code": sales["StockCode"],
        "description": sales["Description"],
        "quantity": sales["Quantity"],
        "invoice_date": sales["InvoiceDate"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "price": sales["Price"],
        "customer_id": sales["Customer ID"].astype("Int64"),     # NULL for guest checkouts
        "country": sales["Country"],
        "revenue": sales["Revenue"],
        "is_large_line": sales["is_large_line"].astype(int),
    }))

    # selected_products.csv is already in sales-rank order
    insert(conn, "selected_products", pd.DataFrame({
        "stock_code": selected["StockCode"],
        "sales_rank": range(1, len(selected) + 1),
        "capped_total": selected["capped_total"],
        "adi": selected["ADI"],
        "cv2": selected["CV2"],
        "pattern": selected["pattern"],
    }))

    insert(conn, "weekly_demand", pd.DataFrame({
        "stock_code": weekly["StockCode"],
        "week": weekly["week"].dt.strftime("%Y-%m-%d"),
        "units": weekly["units"],
    }))

    insert(conn, "cleaning_log", log)

    conn.commit()
    print_counts(conn, RETAIL_DB)
    conn.close()


# ---------------------------------------------------------------- app.db
def build_app_db():
    bundle = joblib.load(BUNDLE_PATH)
    history = bundle["history"]                  # week x product, exactly what the model was trained on

    conn = create_db(APP_DB, "app_schema.sql")

    insert(conn, "products", pd.DataFrame({
        "stock_code": [str(c) for c in bundle["product_names"]],
        "name": list(bundle["product_names"].values()),
    }))

    long = history.rename_axis(index="week", columns="stock_code").stack().rename("units").reset_index()
    insert(conn, "weekly_demand", pd.DataFrame({
        "stock_code": long["stock_code"].astype(str),
        "week": long["week"].dt.strftime("%Y-%m-%d"),
        "units": long["units"].astype(int),
    }))

    insert(conn, "model_info", pd.DataFrame({
        "key": ["trained_until", "test_metrics"],
        "value": [json.dumps(bundle["trained_until"]), json.dumps(bundle["test_metrics"])],
    }))

    conn.commit()
    print_counts(conn, APP_DB)
    conn.close()


if __name__ == "__main__":
    build_retail_db()
    build_app_db()
