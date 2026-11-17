"""
Make forecasts from a saved model bundle.

Used by the notebooks (to check the saved bundles) and by the FastAPI app,
so both produce forecasts in exactly the same way.
"""

import numpy as np
import pandas as pd

from src.features import ORDER_FEATURES, cap_outliers, make_features


def forecast(bundle, history=None, orders=None):
    """
    Forecast the next `horizon` weeks for every product in the bundle.

    bundle  : dict saved by the training notebook (models, fence, history, ...)
    history : optional week x product DataFrame to use instead of bundle["history"]
              (for example, with newer weeks appended).
    orders  : optional week x product order counts, used when the model has order
              features (Part 7); defaults to bundle["orders"].
    Returns a long DataFrame: StockCode, week, h, p10, p50, p90.
    """
    hist = bundle["history"] if history is None else history
    origin = len(hist) - 1
    if any(f in ORDER_FEATURES for f in bundle["features"]):
        orders = bundle["orders"] if orders is None else orders
    else:
        orders = None                                      # Part 4 model: no order features
    rows = make_features(cap_outliers(hist, bundle["fence"]), origin, orders=orders)

    preds = []
    for q in bundle["quantiles"]:
        p = bundle["models"][q].predict(rows[bundle["features"]]) * rows["scale"].to_numpy()
        preds.append(np.clip(p, 0, None))
    p10, p50, p90 = np.sort(np.vstack(preds), axis=0)     # keep p10 <= p50 <= p90

    return pd.DataFrame({
        "StockCode": rows["StockCode"].astype(str).to_numpy(),
        "week": rows["target_week"].to_numpy(),
        "h": rows["h"].to_numpy(),
        "p10": p10.round(1),
        "p50": p50.round(1),
        "p90": p90.round(1),
    })