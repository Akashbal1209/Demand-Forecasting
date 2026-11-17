"""
Statistical checks for comparing forecasts (Part 8):
the Diebold-Mariano test and bootstrap confidence intervals.
"""

import numpy as np
import pandas as pd
from scipy import stats


def diebold_mariano(loss_a, loss_b, max_lag=0):
    """
    Diebold-Mariano test: do forecasts A and B have the same expected loss?

    loss_a, loss_b : 1-D arrays, the loss of each forecast in each time period
    max_lag        : number of lags the loss difference may be autocorrelated
                     (h - 1 for h-step-ahead forecasts)

    Uses the Harvey, Leybourne & Newbold (1997) small-sample correction and a
    t distribution with T - 1 degrees of freedom.
    Returns the mean loss difference (A - B), the test statistic and a two-sided
    p-value. A negative difference means A has the lower loss.
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    T, h = len(d), max_lag + 1
    if T <= 2 * max_lag + 1:
        raise ValueError(f"{T} periods are too few for max_lag={max_lag}")

    dc = d - d.mean()
    gamma = [dc[k:] @ dc[: T - k] / T for k in range(max_lag + 1)]   # autocovariances
    var = (gamma[0] + 2 * sum(gamma[1:])) / T                        # long-run variance of the mean
    if var <= 0:
        raise ValueError("the estimated variance is not positive; use a smaller max_lag")

    hln = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)             # small-sample correction
    stat = d.mean() / np.sqrt(var) * hln
    p_value = 2 * stats.t.sf(abs(stat), df=T - 1)
    return {"mean_diff": d.mean(), "DM": stat, "p_value": p_value}


def bootstrap_ratio(numerators, denominator, n_boot=10_000, seed=0):
    """
    Bootstrap a ratio metric by resampling products with replacement.

    numerators  : dict name -> array of one total per product (e.g. absolute error)
    denominator : array of one total per product (e.g. actual units)

    Each product keeps all its weeks together, so the weeks of one product are
    never split up. The same resampled products are used for every name, so the
    columns can be subtracted to get the distribution of a difference.
    Returns a DataFrame with n_boot rows and one column per name.
    """
    denominator = np.asarray(denominator, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(denominator), size=(n_boot, len(denominator)))
    den = denominator[idx].sum(axis=1)
    return pd.DataFrame({name: np.asarray(num, dtype=float)[idx].sum(axis=1) / den
                         for name, num in numerators.items()})


def bootstrap_wape(actual, forecasts, n_boot=10_000, seed=0):
    """
    Bootstrap distribution of WAPE for one or more forecasts.

    actual    : array (weeks, products)
    forecasts : dict name -> array (weeks, products)
    """
    abs_err = {name: np.abs(f - actual).sum(axis=0) for name, f in forecasts.items()}
    return bootstrap_ratio(abs_err, actual.sum(axis=0), n_boot=n_boot, seed=seed)


def percentile_ci(samples, level=0.95):
    """Percentile confidence interval of bootstrap samples, as (low, high)."""
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(samples, [tail, 100 - tail])
    return low, high
