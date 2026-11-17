"""
Tests for the feature pipeline (src/features.py) and the saved model bundle.

These protect the two things that silently break a forecasting model:
- leakage: a feature that uses data from after the forecast origin
- train/serve skew: features built differently in training and in the API
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.features import (FEATURES, HORIZON, ORDER_FEATURES, PEAK_FEATURES,
                          build_training_set, cap_outliers, make_features)
from src.forecast import forecast

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
BUNDLE_PATH = MODELS_DIR / "forecast_bundle.pkl"            # Part 7 model, used by the app
BUNDLE_V1_PATH = MODELS_DIR / "forecast_bundle_v1.pkl"      # Part 4 model, used by notebooks 04-05


@pytest.fixture(scope="module")
def bundle():
    return joblib.load(BUNDLE_PATH)


@pytest.fixture(scope="module")
def history(bundle):
    return bundle["history"]


def test_no_leakage_from_future_weeks(history):
    """Changing every week after the origin must not change any feature."""
    origin = len(history) - 20
    fake = history.copy()
    fake.iloc[origin + 1:] = 999_999
    pd.testing.assert_frame_equal(make_features(history, origin), make_features(fake, origin))


def test_train_and_serve_features_are_identical(history):
    """The API path (cap + make_features) must build exactly the training features."""
    last = len(history) - 1
    train_rows, fence = build_training_set(history, last)

    origin = last - HORIZON                       # an origin with all 8 horizons in training
    served = make_features(cap_outliers(history.iloc[: last + 1], fence), origin)
    trained = train_rows[train_rows["origin_week"] == history.index[origin]]

    assert len(served) == len(trained) == HORIZON * history.shape[1]
    pd.testing.assert_frame_equal(
        served[FEATURES].reset_index(drop=True),
        trained[FEATURES].reset_index(drop=True),
    )


def test_bundle_has_everything_the_api_needs(bundle):
    for key in ["models", "quantiles", "features", "fence", "history", "orders",
                "product_names", "trained_until"]:
        assert key in bundle
    assert bundle["features"] == FEATURES + ORDER_FEATURES
    assert set(bundle["models"]) == {"p10", "p50", "p90"}
    assert bundle["orders"].index.equals(bundle["history"].index)
    assert bundle["orders"].columns.equals(bundle["history"].columns)


def test_train_and_serve_order_features_are_identical(bundle):
    """Same train/serve check as above, for the app's model with order features."""
    history, orders = bundle["history"], bundle["orders"]
    last = len(history) - 1
    train_rows, fence = build_training_set(history, last, orders=orders)

    origin = last - HORIZON
    served = make_features(cap_outliers(history.iloc[: last + 1], fence), origin, orders=orders)
    trained = train_rows[train_rows["origin_week"] == history.index[origin]]

    cols = bundle["features"]
    pd.testing.assert_frame_equal(served[cols].reset_index(drop=True),
                                  trained[cols].reset_index(drop=True))


def test_old_part4_bundle_still_works():
    """Notebooks 04-05 load the Part 4 model, which has no order features."""
    old = joblib.load(BUNDLE_V1_PATH)
    assert old["features"] == FEATURES
    fc = forecast(old)
    assert len(fc) == HORIZON * old["history"].shape[1]
    assert (fc["p10"] <= fc["p50"]).all() and (fc["p50"] <= fc["p90"]).all()


def test_forecast_output_is_sane(bundle, history):
    fc = forecast(bundle)
    assert len(fc) == HORIZON * history.shape[1]
    assert (fc[["p10", "p50", "p90"]] >= 0).all().all()
    assert (fc["p10"] <= fc["p50"]).all() and (fc["p50"] <= fc["p90"]).all()
    assert np.isfinite(fc[["p10", "p50", "p90"]].to_numpy()).all()


@pytest.fixture(scope="module")
def orders(history):
    """Made-up weekly order counts with the same shape as the history (real ones need data/retail.db)."""
    return (history // 10 + (history > 0)).astype(int)


def test_new_features_do_not_leak(history, orders):
    """Changing future units or future orders must not change the Part 7 features either."""
    origin = len(history) - 20
    fake_y, fake_o = history.copy(), orders.copy()
    fake_y.iloc[origin + 1:] = 999_999
    fake_o.iloc[origin + 1:] = 999_999
    cols = FEATURES + PEAK_FEATURES + ORDER_FEATURES
    pd.testing.assert_frame_equal(make_features(history, origin, orders=orders)[cols],
                                  make_features(fake_y, origin, orders=fake_o)[cols])


def test_orders_must_match_products(history, orders):
    with pytest.raises(ValueError):
        make_features(history, len(history) - 1, orders=orders.iloc[:, ::-1])
