"""
Feature engineering for weekly demand forecasting.

This file is shared by the training notebook and the FastAPI app, so the
model always sees features built in exactly the same way.

Setup ("direct" forecasting):
    - Forecast origin T = the last week whose sales we know.
    - Horizon h = 1 ... 8 weeks ahead. Target week = T + h.
    - Every feature is computed ONLY from weeks 0 ... T (no future data).
    - Level-type features are divided by the product's recent level
      (its 13-week average, called "scale"), so one global model can
      serve products that sell 50 units a week and products that sell 2,000.
      The model predicts units / scale; multiply by scale to get units.
"""

import numpy as np
import pandas as pd
import holidays

HORIZON = 8         # weeks ahead to forecast
SEASON = 52         # weeks in a year
MIN_HISTORY = 13    # weeks needed before the first forecast origin

FEATURES = [
    "h",
    "lag_1", "lag_2", "lag_3", "lag_4",
    "rmean_4", "rmean_8", "rstd_8",
    "ly_same", "ly_smooth", "yoy_ratio", "season_ratio", "log_scale",
    "week_of_year", "month", "n_holidays", "weeks_to_xmas",
    "is_xmas_week", "is_newyear_week",
    "product",
]
CATEGORICAL = ["product"]


# calendar
def week_calendar(week_starts):
    """Calendar and UK (England) holiday features for weeks starting on Monday."""
    week_starts = pd.DatetimeIndex(week_starts)
    years = range(week_starts.year.min() - 1, week_starts.year.max() + 2)
    uk = holidays.country_holidays("GB", subdiv="ENG", years=years)
    hol = pd.DatetimeIndex(sorted(uk.keys()))

    week_ends = week_starts + pd.Timedelta(days=6)
    cal = pd.DataFrame(index=week_starts)
    cal["week_of_year"] = week_starts.isocalendar().week.astype(int).to_numpy()
    cal["month"] = (week_starts + pd.Timedelta(days=3)).month  # month of the week's Thursday
    cal["n_holidays"] = [int(((hol >= s) & (hol <= e)).sum()) for s, e in zip(week_starts, week_ends)]

    # weeks until the next Christmas Day (0 = Christmas falls in this week)
    xmas = pd.to_datetime([f"{y}-12-25" for y in week_starts.year])
    days = (xmas - week_starts).days.to_numpy()
    next_xmas = pd.to_datetime([f"{y + 1}-12-25" for y in week_starts.year])
    days = np.where(days < 0, (next_xmas - week_starts).days, days)
    cal["weeks_to_xmas"] = days // 7

    def contains(month, day):
        d = pd.to_datetime([f"{y}-{month:02d}-{day:02d}" for y in week_starts.year])
        return ((d >= week_starts) & (d <= week_ends)).astype(int)

    cal["is_xmas_week"] = contains(12, 25)
    cal["is_newyear_week"] = contains(12, 31)
    return cal


#  outliers
def outlier_fence(Y_train):
    """Per-product upper fence Q3 + 3 * IQR, from training weeks only."""
    q1, q3 = Y_train.quantile(0.25), Y_train.quantile(0.75)
    return q3 + 3 * (q3 - q1)


def cap_outliers(Y, fence):
    """Cap each product's weekly units at its fence (removes one-off bulk orders)."""
    return Y.clip(upper=fence, axis=1)


#  features
def _window_mean(hist, start, end):
    """Mean of rows start..end (inclusive); NaN if the window is out of range."""
    if start < 0 or end >= len(hist):
        return np.full(hist.shape[1], np.nan)
    return hist[start : end + 1].mean(axis=0)


def make_features(Y, origin, horizons=range(1, HORIZON + 1)):
    """
    Feature rows for forecasting weeks origin+1 ... origin+H.

    Y      : DataFrame, week x product (weekly DatetimeIndex, columns = StockCode).
    origin : integer position of the last known week.
    Only rows 0 ... origin of Y are read, so future values can't leak in.
    """
    if origin < MIN_HISTORY - 1:
        raise ValueError(f"origin must be >= {MIN_HISTORY - 1}")

    hist = Y.iloc[: origin + 1].to_numpy(dtype=float)
    T = origin
    products = Y.columns

    rmean_4 = _window_mean(hist, T - 3, T)
    rmean_8 = _window_mean(hist, T - 7, T)
    rmean_13 = _window_mean(hist, T - 12, T)
    rstd_8 = hist[T - 7 : T + 1].std(axis=0)
    scale = np.maximum(rmean_13, 1.0)

    # same 13 weeks one year earlier -> year-over-year change in level
    ly_level = _window_mean(hist, T - SEASON - 12, T - SEASON)
    with np.errstate(divide="ignore", invalid="ignore"):
        yoy_ratio = np.where(ly_level > 0, rmean_13 / ly_level, np.nan)
    yoy_ratio = np.clip(yoy_ratio, 0, 5)

    base = {
        "lag_1": hist[T] / scale,
        "lag_2": hist[T - 1] / scale,
        "lag_3": hist[T - 2] / scale,
        "lag_4": hist[T - 3] / scale,
        "rmean_4": rmean_4 / scale,
        "rmean_8": rmean_8 / scale,
        "rstd_8": rstd_8 / scale,
        "yoy_ratio": yoy_ratio,
        "log_scale": np.log1p(scale),
    }

    origin_week = Y.index[T]
    target_weeks = pd.DatetimeIndex([origin_week + pd.Timedelta(weeks=h) for h in horizons])
    cal = week_calendar(target_weeks)

    frames = []
    for h, tw in zip(horizons, target_weeks):
        lw = T + h - SEASON                       # same week last year
        ly_same = hist[lw] if 0 <= lw <= T else np.full(len(products), np.nan)
        ly_smooth = _window_mean(hist, lw - 2, lw + 2)      # 5 weeks around it
        with np.errstate(divide="ignore", invalid="ignore"):
            season_ratio = np.where(ly_level > 0, ly_smooth / ly_level, np.nan)
        season_ratio = np.clip(season_ratio, 0, 5)

        f = pd.DataFrame(base, index=products)
        f["ly_same"] = ly_same / scale
        f["ly_smooth"] = ly_smooth / scale
        f["season_ratio"] = season_ratio      # last year's week vs last year's level
        f["h"] = h
        for col in cal.columns:
            f[col] = cal.loc[tw, col]
        f["scale"] = scale
        f["origin_week"] = origin_week
        f["target_week"] = tw
        frames.append(f)

    out = pd.concat(frames).rename_axis("StockCode").reset_index()
    out["product"] = pd.Categorical(out["StockCode"].astype(str), categories=[str(c) for c in products])
    return out


def build_training_set(Y, last_train, first_origin=MIN_HISTORY - 1):
    """
    Training rows for a model that may only use weeks 0 ... last_train.

    1. Outlier fence from the training weeks only; cap the training series.
    2. For every origin T, keep horizons whose target week T+h <= last_train.
    3. Target = capped units in week T+h divided by the product's scale.
    Returns (rows, fence).
    """
    Y_train = Y.iloc[: last_train + 1]
    fence = outlier_fence(Y_train)
    Y_cap = cap_outliers(Y_train, fence)

    frames = []
    for T in range(first_origin, last_train):
        hs = [h for h in range(1, HORIZON + 1) if T + h <= last_train]
        f = make_features(Y_cap, T, hs)
        actual = Y_cap.iloc[[T + h for h in hs]].to_numpy()          # (len(hs), products)
        f["y"] = actual.reshape(-1) / f["scale"].to_numpy()
        frames.append(f)
    return pd.concat(frames, ignore_index=True), fence