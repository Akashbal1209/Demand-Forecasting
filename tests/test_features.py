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

from src.features import (FEATURES, HORIZON, build_training_set, cap_outliers,
                          make_features)
from src.forecast import forecast

BUNDLE_PATH = Path(__file__).resolve().parent.parent / "models" / "forecast_bundle.pkl"


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
    for key in ["models", "quantiles", "features", "fence", "history", "product_names", "trained_until"]:
        assert key in bundle
    assert bundle["features"] == FEATURES
    assert set(bundle["models"]) == {"p10", "p50", "p90"}


def test_forecast_output_is_sane(bundle, history):
    fc = forecast(bundle)
    assert len(fc) == HORIZON * history.shape[1]
    assert (fc[["p10", "p50", "p90"]] >= 0).all().all()
    assert (fc["p10"] <= fc["p50"]).all() and (fc["p50"] <= fc["p90"]).all()
    assert np.isfinite(fc[["p10", "p50", "p90"]].to_numpy()).all()
