"""Tests for the FastAPI app (app/main.py)."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def first_code(client):
    return client.get("/products").json()[0]["stock_code"]


def test_home_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["n_products"] == 100


def test_products(client):
    r = client.get("/products")
    assert r.status_code == 200
    assert len(r.json()) == 100


def test_forecast_ok(client, first_code):
    r = client.post("/forecast", json={"stock_code": first_code, "weeks": 8})
    assert r.status_code == 200
    body = r.json()
    assert len(body["forecast"]) == 8
    for w in body["forecast"]:
        assert w["p10"] <= w["p50"] <= w["p90"]
        assert w["stock_target"] == w["p90"]
        assert w["safety_stock"] == pytest.approx(w["p90"] - w["p50"], abs=0.11)
    assert len(body["history"]) == 26


def test_forecast_fewer_weeks(client, first_code):
    r = client.post("/forecast", json={"stock_code": first_code, "weeks": 3})
    assert r.status_code == 200
    assert [w["h"] for w in r.json()["forecast"]] == [1, 2, 3]


def test_unknown_product_returns_404(client):
    r = client.post("/forecast", json={"stock_code": "99999"})
    assert r.status_code == 404


@pytest.mark.parametrize("weeks", [0, 9])
def test_weeks_out_of_range_returns_422(client, first_code, weeks):
    r = client.post("/forecast", json={"stock_code": first_code, "weeks": weeks})
    assert r.status_code == 422
