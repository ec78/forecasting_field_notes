"""Tests for the Post 05 null-testing pieces that are easy to get subtly wrong:
the SPA sign convention, causal volatility standardisation, return decoders,
ridge lookahead, the MAE-ratio interval, and the planted-signal process."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation import rolling_origin_evaluate
from src.null_testing import (
    NO_CHANGE,
    add_return_columns,
    cell_table,
    chronos_price_forecast,
    chronos_return_forecast,
    chronos_standardised_forecast,
    drift_forecast,
    ewma_volatility,
    family_loss_differentials,
    mae_ratio_with_ci,
    no_change_forecast,
    oracle_forecast,
    planted_sign_series,
    ridge_design,
    ridge_features,
    spa_test,
    valid_positions,
)


def _logp(n=300, seed=0):
    rng = np.random.default_rng(seed)
    values = np.cumsum(rng.normal(0, 0.01, n)) + np.log(100)
    return pd.Series(values, index=pd.bdate_range("2020-01-01", periods=n))


# --- SPA sign convention ----------------------------------------------------

def test_spa_small_p_when_a_cell_beats_no_change():
    rng = np.random.default_rng(1)
    t = 400
    good = -0.5 + rng.normal(0, 0.2, t)      # loss - loss_no_change < 0: beats no-change
    useless = rng.normal(0, 0.2, t)
    diffs = pd.DataFrame({"good": good, "useless": useless})
    assert spa_test(diffs, block_size=5, reps=500)["p_consistent"] < 0.01


def test_spa_large_p_when_every_cell_loses():
    rng = np.random.default_rng(2)
    t = 400
    diffs = pd.DataFrame({"a": 0.5 + rng.normal(0, 0.2, t), "b": 0.3 + rng.normal(0, 0.2, t)})
    result = spa_test(diffs, block_size=5, reps=500)
    # Hansen's floored statistic is 0 when no cell has a positive mean d, so p = 1
    # (arch's unfloored p is below 1 here, which is why the floor is applied).
    assert result["statistic_is_zero"]
    assert result["p_consistent"] == 1.0
    assert result["p_consistent_arch_raw"] < 1.0


# --- MAE ratio and SMI --------------------------------------------------------

def test_mae_ratio_interval_and_smi():
    loss_no_change = np.full(200, 1.0)
    better = np.full(200, 0.9)
    stats = mae_ratio_with_ci(better, loss_no_change, block_length=5, n_boot=200)
    assert stats["mae_ratio"] == pytest.approx(0.9)
    assert stats["ratio_high"] == pytest.approx(0.9)   # zero-variance difference
    assert stats["meets_smi"]

    tiny_gain = np.full(200, 0.998)                    # below 1 but above the 0.995 bar
    assert not mae_ratio_with_ci(tiny_gain, loss_no_change, 5, n_boot=200)["meets_smi"]


def test_harness_on_log_price_gives_return_losses():
    logp = _logp(120)
    res = rolling_origin_evaluate(logp, no_change_forecast, NO_CHANGE, [1, 5], 100)
    scored = add_return_columns(res)
    # no-change forecasts a zero return, so its loss equals |y|
    assert np.allclose(scored["f"], 0.0)
    assert np.allclose(scored["loss"], scored["loss_no_change"])


def test_family_differentials_align_on_common_origins():
    logp = _logp(140)
    a = rolling_origin_evaluate(logp, drift_forecast, "Drift", [1, 5], 100,
                                model_kwargs={"context_length": 50})
    nc = rolling_origin_evaluate(logp, no_change_forecast, NO_CHANGE, [1, 5], 100)
    diffs = family_loss_differentials(pd.concat([a, nc]), [("Drift", 1), ("Drift", 5)])
    assert diffs.shape == (len(a) // 2, 2)
    assert not diffs.isna().any().any()
    table = cell_table(pd.concat([a, nc]), n_boot=100)
    assert set(table["model"]) == {"Drift"}


# --- causal volatility and decoders ---------------------------------------------

def test_ewma_sigma_is_causal():
    r = np.random.default_rng(3).normal(0, 0.01, 100)
    sigma, _ = ewma_volatility(r)
    bumped = r.copy()
    bumped[60] = 0.5
    sigma_bumped, _ = ewma_volatility(bumped)
    assert np.allclose(sigma[:61], sigma_bumped[:61])  # sigma[60] must not see r[60]
    assert sigma_bumped[61] > sigma[61]


def _echo_base(values):
    """Fake base model: forecasts each step as a fixed known value."""
    def base_fn(series, horizon, **_):
        return {"forecast": np.asarray(values[:horizon], dtype=float)}
    return base_fn


def test_return_decoder_sums_per_step_medians():
    logp = _logp(50)
    steps = [0.01, -0.02, 0.03]
    path = chronos_return_forecast(logp, 3, base_fn=_echo_base(steps))
    assert np.allclose(path, logp.iloc[-1] + np.cumsum(steps))


def test_standardised_decoder_rescales_by_origin_volatility():
    logp = _logp(80)
    _, sigma_next = ewma_volatility(np.diff(logp.to_numpy()))
    steps = [1.0, 1.0]
    path = chronos_standardised_forecast(logp, 2, base_fn=_echo_base(steps))
    assert np.allclose(path, logp.iloc[-1] + sigma_next * np.cumsum(steps))


def test_price_decoder_takes_logs():
    logp = _logp(50)
    path = chronos_price_forecast(logp, 2, base_fn=_echo_base([105.0, 110.0]))
    assert np.allclose(path, np.log([105.0, 110.0]))


def test_drift_uses_only_context():
    logp = pd.Series(np.concatenate([np.zeros(100), np.arange(10) * 0.01]))
    path = drift_forecast(logp, 3, context_length=10)
    assert np.allclose(np.diff(np.concatenate([[logp.iloc[-1]], path])), 0.01)


# --- ridge lookahead -----------------------------------------------------------

def test_ridge_features_ignore_the_future():
    values = _logp(100).to_numpy()
    changed = values.copy()
    changed[60:] += 1.0                      # perturb everything from index 60 on
    assert np.allclose(ridge_features(values, 60), ridge_features(changed, 60))


def test_ridge_targets_and_valid_positions():
    values = _logp(100).to_numpy()
    positions = valid_positions(30, last_target_index=80, horizon=5)
    assert positions[-1] - 1 + 5 == 80      # last target exactly at the boundary
    _, y = ridge_design(values, positions, 5)
    assert y[0] == pytest.approx(values[30 - 1 + 5] - values[30 - 1])


# --- planted signal -------------------------------------------------------------

def test_planted_series_keeps_magnitudes_and_persistence():
    rng = np.random.default_rng(4)
    mags = np.abs(rng.normal(0, 0.01, 20_000)) + 1e-6
    series = planted_sign_series(mags, delta=0.2, rng=rng)
    r = np.diff(series.to_numpy())
    assert np.allclose(np.abs(r), mags)
    stay = np.mean(np.sign(r[1:]) == np.sign(r[:-1]))
    assert stay == pytest.approx(0.7, abs=0.02)


def test_oracle_forecast_matches_markov_expectation():
    returns = [0.01, -0.02, 0.03]                                    # last return positive
    logp = pd.Series(np.log(100) + np.concatenate([[0.0], np.cumsum(returns)]))
    path = oracle_forecast(logp, 2, delta=0.25, magnitude_window=3)
    m = np.mean([0.01, 0.02, 0.03])
    expected = logp.iloc[-1] + np.cumsum([0.5 * m, 0.25 * m])
    assert np.allclose(path, expected)
