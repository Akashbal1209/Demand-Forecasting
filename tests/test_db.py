"""
Tests for the SQLite layer (src/db.py, sql/).

- data/app.db must hold exactly the data the model was trained on,
  so the API gives the same forecasts as the saved bundle.
- The weekly demand SQL query must follow the same rules as notebook 02.
"""

import sqlite3
from contextlib import closing
from pathlib import Path

import joblib
import pandas as pd
import pytest

from src.db import (APP_DB, SQL_DIR, connect, load_history, load_model_info,
                    load_orders, load_product_names, run_query)
from src.forecast import forecast

BUNDLE_PATH = Path(__file__).resolve().parent.parent / "models" / "forecast_bundle.pkl"


@pytest.fixture(scope="module")
def bundle():
    return joblib.load(BUNDLE_PATH)


@pytest.fixture(scope="module")
def app_conn():
    with closing(connect(APP_DB)) as conn:
        yield conn


# ---------------------------------------------------------------- app.db
def test_history_matches_bundle(app_conn, bundle):
    history = load_history(app_conn)
    pd.testing.assert_frame_equal(history, bundle["history"], check_column_type=False)


def test_orders_match_bundle(app_conn, bundle):
    orders = load_orders(app_conn)
    pd.testing.assert_frame_equal(orders, bundle["orders"], check_column_type=False,
                                  check_dtype=False)


def test_forecasts_match_bundle(app_conn, bundle):
    from_db = forecast(bundle, history=load_history(app_conn), orders=load_orders(app_conn))
    pd.testing.assert_frame_equal(from_db, forecast(bundle))


def test_products_and_model_info_match_bundle(app_conn, bundle):
    assert load_product_names(app_conn) == {str(c): n for c, n in bundle["product_names"].items()}
    info = load_model_info(app_conn)
    assert info["trained_until"] == bundle["trained_until"]
    assert info["test_metrics"] == bundle["test_metrics"]


def test_app_db_is_opened_read_only(app_conn):
    with pytest.raises(sqlite3.OperationalError):
        app_conn.execute("DELETE FROM products")


# ---------------------------------------------------------------- weekly demand query
@pytest.fixture
def tiny_db():
    """A small in-memory copy of the retail schema with a few hand-made sales."""
    conn = sqlite3.connect(":memory:")
    conn.executescript((SQL_DIR / "schema.sql").read_text(encoding="utf-8"))
    conn.executemany("INSERT INTO products VALUES (?, ?)",
                     [("10001", "A"), ("10002", "B"), ("10003", "C")])
    conn.executemany("INSERT INTO selected_products VALUES (?, ?, 0, 1, 0, 'smooth')",
                     [("10001", 1), ("10002", 2)])
    sales = [  # (stock_code, quantity, invoice_date)
        ("10001", 7, "2010-01-05 10:00:00"),   # Tue, first (partial) week -> dropped
        ("10001", 5, "2010-01-11 09:00:00"),   # Mon, week 2010-01-11
        ("10001", 3, "2010-01-17 18:00:00"),   # Sun, same week 2010-01-11
        ("10002", 4, "2010-01-19 12:00:00"),   # Tue, week 2010-01-18
        ("10003", 50, "2010-01-26 12:00:00"),  # product not selected -> ignored
        ("10001", 2, "2010-02-02 12:00:00"),   # Tue, week 2010-02-01
        ("10002", 9, "2010-02-08 12:00:00"),   # Mon, last (partial) week -> dropped
    ]
    conn.executemany(
        "INSERT INTO sales (invoice, stock_code, description, quantity, invoice_date, "
        "price, country, revenue, is_large_line) VALUES ('1', ?, 'x', ?, ?, 1, 'UK', 1, 0)",
        sales,
    )
    yield conn
    conn.close()


def test_weekly_demand_query(tiny_db):
    weekly = run_query(tiny_db, "weekly_demand")
    weeks = ["2010-01-11", "2010-01-18", "2010-01-25", "2010-02-01"]

    assert list(weekly.columns) == ["stock_code", "week", "units"]
    assert weekly["week"].unique().tolist() == weeks          # Mondays, partial weeks dropped
    units = weekly.pivot(index="week", columns="stock_code", values="units")
    assert units["10001"].tolist() == [8, 0, 0, 2]             # Mon + Sun summed, gaps filled with 0
    assert units["10002"].tolist() == [0, 4, 0, 0]
    assert "10003" not in units.columns                       # only selected products


def test_schema_rejects_bad_sales(tiny_db):
    with pytest.raises(sqlite3.IntegrityError):
        tiny_db.execute(
            "INSERT INTO sales (invoice, stock_code, description, quantity, invoice_date, "
            "price, country, revenue, is_large_line) "
            "VALUES ('1', '10001', 'x', -3, '2010-01-11', 1, 'UK', 1, 0)"   # negative quantity
        )
    with pytest.raises(sqlite3.IntegrityError):
        tiny_db.execute(
            "INSERT INTO sales (invoice, stock_code, description, quantity, invoice_date, "
            "price, country, revenue, is_large_line) "
            "VALUES ('1', '99999', 'x', 3, '2010-01-11', 1, 'UK', 1, 0)"    # unknown product
        )
