"""
SQLite helpers.

The project keeps two SQLite databases, both built by scripts/build_db.py:
    data/retail.db : all clean sales lines, used by the SQL analysis notebook (06)
    data/app.db    : the 100 forecast products, their weekly units and orders, and the
                     model's test metrics, read by the FastAPI app

sqlite3 is part of Python's standard library, so no database server is needed:
each database is a single file.
"""

import json
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SQL_DIR = ROOT / "sql"
QUERY_DIR = SQL_DIR / "queries"
RETAIL_DB = ROOT / "data" / "retail.db"
APP_DB = ROOT / "data" / "app.db"


def connect(path, read_only=True):
    """
    Open a SQLite database.

    read_only=True opens the file in read-only mode, so a query can never change
    the data by mistake. It also fails loudly if the file does not exist,
    instead of silently creating an empty database.
    """
    path = Path(path)
    if read_only:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Build it with: python scripts/build_db.py")
        conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    else:
        conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# saved queries (sql/queries/*.sql)
def read_query(name):
    """Text of a saved query, e.g. read_query("top_products")."""
    return (QUERY_DIR / f"{name}.sql").read_text(encoding="utf-8")


def run_query(conn, name, **params):
    """Run a saved query with named parameters and return the result as a DataFrame."""
    return pd.read_sql(read_query(name), conn, params=params or None)


# data for the forecast app (data/app.db)
def load_product_names(conn):
    """{stock_code: product name} for the forecast products."""
    rows = conn.execute("SELECT stock_code, name FROM products ORDER BY stock_code").fetchall()
    return dict(rows)


def _load_weekly(conn, table, value):
    """A weekly table as a week x product DataFrame (rows = weeks, columns = stock codes)."""
    df = pd.read_sql(
        f"""
        SELECT week, stock_code, {value}
        FROM {table}
        ORDER BY stock_code, week
        """,
        conn,
        parse_dates=["week"],
    )
    wide = df.pivot(index="week", columns="stock_code", values=value)
    wide.columns.name = "StockCode"
    return wide


def load_history(conn):
    """
    Weekly units as a week x product table, the same shape as bundle["history"]
    and what src/forecast.py expects.
    """
    return _load_weekly(conn, "weekly_demand", "units")


def load_orders(conn):
    """Orders (distinct invoices) per week, same shape as load_history()."""
    return _load_weekly(conn, "weekly_orders", "orders")


def load_model_info(conn):
    """Facts about the saved model, e.g. {"trained_until": "2011-11-28", "test_metrics": {...}}."""
    rows = conn.execute("SELECT key, value FROM model_info").fetchall()
    return {key: json.loads(value) for key, value in rows}
