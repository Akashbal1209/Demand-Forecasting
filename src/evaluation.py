"""
Evaluation metrics and baseline forecasts, shared by all notebooks
so every model is scored in exactly the same way.
"""

import numpy as np

from src.features import HORIZON, outlier_fence


def evaluate(actual, forecast, train):
    """
    actual, forecast : arrays of shape (weeks, products)
    train            : DataFrame of the training weeks (week x product)
    """
    err = forecast - actual
    abs_err = np.abs(err)

    # MASE scale: in-sample error of "same as last week", per product
    scale = train.diff().abs().iloc[1:].mean().to_numpy()

    # bulk weeks: actual above the training-data outlier fence
    normal = actual <= outlier_fence(train).to_numpy()

    return {
        "MAE": abs_err.mean(),
        "WAPE": abs_err.sum() / actual.sum(),
        "Bias": err.sum() / actual.sum(),
        "MASE": np.nanmean(abs_err.mean(axis=0) / scale),
        "WAPE (no bulk weeks)": abs_err[normal].sum() / actual[normal].sum(),
        "bulk weeks": int((~normal).sum()),
    }


def moving_average_forecast(Y, origin, k=13, H=HORIZON):
    """Best baseline from Part 3: every future week = mean of the last k weeks."""
    level = Y.iloc[origin - k + 1 : origin + 1].mean().to_numpy()
    return np.tile(level, (H, 1))