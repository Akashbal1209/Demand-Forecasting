"""
Tests for the statistical checks (src/stats.py) used in notebook 08.

The Diebold-Mariano test is checked against a known special case (with no
autocorrelation it is a paired t-test), and the bootstrap against simple cases
where the answer is known.
"""

import numpy as np
import pytest
from scipy import stats

from src.stats import bootstrap_wape, diebold_mariano, percentile_ci


@pytest.fixture(scope="module")
def losses():
    rng = np.random.default_rng(1)
    loss_b = rng.gamma(2.0, 10.0, size=40)
    loss_a = loss_b + rng.normal(-1.0, 3.0, size=40)       # A slightly better on average
    return loss_a, loss_b


def test_dm_without_lags_is_a_paired_t_test(losses):
    loss_a, loss_b = losses
    r = diebold_mariano(loss_a, loss_b, max_lag=0)
    t = stats.ttest_rel(loss_a, loss_b)
    assert r["DM"] == pytest.approx(t.statistic)
    assert r["p_value"] == pytest.approx(t.pvalue)
    assert r["mean_diff"] == pytest.approx(np.mean(loss_a - loss_b))


def test_dm_finds_a_clearly_better_forecast(losses):
    _, loss_b = losses
    r = diebold_mariano(loss_b * 0.5, loss_b, max_lag=3)
    assert r["DM"] < 0 and r["p_value"] < 0.001


def test_dm_needs_enough_periods():
    with pytest.raises(ValueError):
        diebold_mariano(np.arange(8.0), np.arange(8.0)[::-1], max_lag=7)


@pytest.fixture(scope="module")
def panel():
    rng = np.random.default_rng(2)
    actual = rng.poisson(50, size=(8, 30)).astype(float)
    good = actual + rng.normal(0, 5, size=actual.shape)
    bad = actual + rng.normal(0, 15, size=actual.shape)
    return actual, good, bad


def test_bootstrap_wape_is_centred_on_the_test_wape(panel):
    actual, good, _ = panel
    point = np.abs(good - actual).sum() / actual.sum()
    boot = bootstrap_wape(actual, {"good": good}, n_boot=2_000)
    low, high = percentile_ci(boot["good"])
    assert low < point < high
    assert boot["good"].mean() == pytest.approx(point, rel=0.05)


def test_paired_bootstrap_uses_the_same_products(panel):
    actual, good, bad = panel
    boot = bootstrap_wape(actual, {"a": good, "b": good.copy(), "bad": bad}, n_boot=2_000)
    assert (boot["a"] == boot["b"]).all()                   # same forecast -> difference exactly 0
    assert (boot["a"] < boot["bad"]).all()


def test_bootstrap_is_reproducible(panel):
    actual, good, _ = panel
    boot_a = bootstrap_wape(actual, {"good": good}, n_boot=500, seed=7)
    boot_b = bootstrap_wape(actual, {"good": good}, n_boot=500, seed=7)
    assert boot_a.equals(boot_b)


def test_percentile_ci():
    low, high = percentile_ci(np.arange(1001), level=0.9)
    assert (low, high) == pytest.approx((50.0, 950.0))
