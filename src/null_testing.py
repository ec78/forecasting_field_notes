"""Forecasters and tests for checking return-forecast skill against a no-change null.

Built for Post 05 (``notebooks/05-declare-the-null/``), which mirrors the
preregistered design of Upadhyay (2026, arXiv:2611.12141) on daily SPY. Every
forecaster here works on a **log-price** series and returns a log-price path,
so it plugs straight into ``src.evaluation.rolling_origin_evaluate``. Scored
that way, ``|actual - forecast|`` *is* the absolute error on the h-step log
return, and the no-change forecast is the naive forecast.

Pieces:

- return-space decoders around a frozen Chronos-2 (R0 price levels, R1 log
  returns, R2 EWMA-volatility-standardised returns);
- the ingredient controls: a drift forecast and a frozen return ridge;
- scoring against no-change: MAE ratio + "smallest meaningful improvement"
  (SMI), Hansen's SPA test over a family of cells, and the registered
  descriptive checks (price-level correlation, directional accuracy vs. the
  up-day base rate, forecast dispersion);
- a planted-signal generator and oracle for the detection-limit check.
"""

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from src.evaluation import paired_block_bootstrap
from src.models import chronos_zero_shot_forecast

NO_CHANGE = "No-change"
SMI_RATIO = 0.995


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _values(train):
    return np.asarray(train, dtype=float)


def _base_forecast(base_fn, series, horizon, base_kwargs):
    output = base_fn(series, horizon=horizon, **(base_kwargs or {}))
    forecast = output["forecast"] if isinstance(output, dict) else output
    return np.asarray(forecast, dtype=float).reshape(-1)[:horizon]


def ewma_volatility(returns, lam=0.94, init_window=20):
    """Causal EWMA volatility (RiskMetrics recursion).

    Returns ``(sigma, sigma_next)``: ``sigma[k]`` uses squared returns strictly
    before ``k`` (so ``returns[k] / sigma[k]`` is a causal standardisation),
    and ``sigma_next`` uses every return given -- the one-step-ahead
    volatility at the origin. The recursion is seeded with the mean squared
    return of the first ``init_window`` observations, which are all in the
    past of any origin this is called at.
    """
    r = np.asarray(returns, dtype=float)
    if len(r) < 2:
        raise ValueError("need at least two returns.")
    var = np.empty(len(r) + 1)
    var[0] = np.mean(r[: max(1, min(init_window, len(r)))] ** 2)
    for k in range(len(r)):
        var[k + 1] = lam * var[k] + (1 - lam) * r[k] ** 2
    sigma = np.sqrt(var)
    return sigma[:-1], float(sigma[-1])


# ---------------------------------------------------------------------------
# Forecasters (log-price in, log-price path out)
# ---------------------------------------------------------------------------

def no_change_forecast(train, horizon):
    return np.full(horizon, _values(train)[-1])


def drift_forecast(train, horizon, context_length=512):
    """Last log price plus h times the mean daily log return over the context.

    Uses the same ``context_length`` prices Chronos-2 is shown, so it is an
    ingredient control: the drift is information the pretrained model also has.
    """
    logp = _values(train)[-context_length:]
    mu = np.mean(np.diff(logp))
    return logp[-1] + mu * np.arange(1, horizon + 1)


def chronos_price_forecast(train, horizon, max_context_length=512,
                           base_fn=chronos_zero_shot_forecast, base_kwargs=None):
    """R0: Chronos-2 on price levels, decoded as the log of its median path."""
    price = pd.Series(np.exp(_values(train)), index=getattr(train, "index", None))
    kwargs = {"max_context_length": max_context_length, **(base_kwargs or {})}
    path = _base_forecast(base_fn, price, horizon, kwargs)
    if np.any(path <= 0):
        raise ValueError("price forecast must be positive to take logs.")
    return np.log(path)


def chronos_return_forecast(train, horizon, max_context_length=512,
                            base_fn=chronos_zero_shot_forecast, base_kwargs=None):
    """R1: Chronos-2 on daily log returns; the h-step return is the sum of per-step medians."""
    logp = _values(train)
    returns = pd.Series(np.diff(logp))
    kwargs = {"max_context_length": max_context_length, **(base_kwargs or {})}
    steps = _base_forecast(base_fn, returns, horizon, kwargs)
    return logp[-1] + np.cumsum(steps)


def chronos_standardised_forecast(train, horizon, max_context_length=512, lam=0.94,
                                  base_fn=chronos_zero_shot_forecast, base_kwargs=None):
    """R2: Chronos-2 on causally volatility-standardised returns.

    ``z_k = r_k / sigma_k`` with an EWMA sigma of *past* squared returns; the
    forecast is decoded with ``sigma_next`` (the one-step-ahead volatility at
    the origin) held fixed over the horizon, as in Upadhyay (2026).
    """
    logp = _values(train)
    returns = np.diff(logp)
    sigma, sigma_next = ewma_volatility(returns, lam=lam)
    z = pd.Series(returns / sigma)
    kwargs = {"max_context_length": max_context_length, **(base_kwargs or {})}
    steps = _base_forecast(base_fn, z, horizon, kwargs)
    return logp[-1] + np.cumsum(sigma_next * steps)


# ---------------------------------------------------------------------------
# Frozen return ridge (the "same information, cheap model" comparator)
# ---------------------------------------------------------------------------

RIDGE_LAGS = 20


def ridge_features(logp_values, position):
    """Features at an origin with ``position`` observations available.

    Last 20 daily log returns (most recent first) plus their 5- and 20-day
    sums, as registered. The 20-day sum is a linear combination of the lags,
    which ridge tolerates; it is kept because the plan names it. Uses only
    ``logp_values[:position]``.
    """
    r = np.diff(np.asarray(logp_values, dtype=float)[:position])
    if len(r) < RIDGE_LAGS:
        raise ValueError(f"need at least {RIDGE_LAGS + 1} observations.")
    window = r[-RIDGE_LAGS:]
    return np.concatenate([window[::-1], [window[-5:].sum(), window.sum()]])


def ridge_design(logp_values, positions, horizon):
    """(X, y) over origins ``positions``; y is the h-step log return after each origin."""
    logp_values = np.asarray(logp_values, dtype=float)
    X = np.array([ridge_features(logp_values, p) for p in positions])
    y = np.array([logp_values[p - 1 + horizon] - logp_values[p - 1] for p in positions])
    return X, y


def fit_return_ridge(logp_values, fit_positions, val_positions, horizon, alphas):
    """Fit a standardised ridge per alpha on ``fit_positions``; pick alpha by validation MAE.

    Returns ``(model, alpha, val_scores)``. The returned model is fit on the
    fit positions only and is then frozen (the plan does not refit on
    validation). Callers must pass positions whose targets lie inside the
    window they belong to.
    """
    X_val, y_val = ridge_design(logp_values, val_positions, horizon)
    scores, models = {}, {}
    for alpha in alphas:
        model = fit_ridge_fixed_alpha(logp_values, fit_positions, horizon, alpha)
        scores[alpha] = float(np.mean(np.abs(y_val - model.predict(X_val))))
        models[alpha] = model
    best = min(scores, key=scores.get)
    return models[best], best, scores


def fit_ridge_fixed_alpha(logp_values, positions, horizon, alpha):
    """Standardised ridge of the h-step return on ``ridge_features``, at a given alpha."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = ridge_design(logp_values, positions, horizon)
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(X, y)


def valid_positions(first_position, last_target_index, horizon, min_position=RIDGE_LAGS + 1):
    """Origin positions from ``first_position`` whose h-step target index is <= ``last_target_index``."""
    start = max(first_position, min_position)
    return list(range(start, last_target_index - horizon + 2))


class RidgeReturnForecaster:
    """Harness-compatible wrapper around frozen per-horizon return ridges.

    Only the fitted horizons get a value; other path entries are NaN (the
    harness reads off just the horizons it scores).
    """

    def __init__(self, models):
        self.models = dict(models)

    def __call__(self, train, horizon):
        logp = _values(train)
        features = ridge_features(logp, len(logp)).reshape(1, -1)
        path = np.full(horizon, np.nan)
        for h, model in self.models.items():
            if h <= horizon:
                path[h - 1] = logp[-1] + float(model.predict(features)[0])
        return path


# ---------------------------------------------------------------------------
# Scoring against no-change
# ---------------------------------------------------------------------------

def add_return_columns(results):
    """Add realised return ``y``, forecast return ``f`` and the two losses to harness output."""
    out = results.copy()
    out["y"] = out["actual"] - out["origin_actual"]
    out["f"] = out["forecast"] - out["origin_actual"]
    out["loss"] = (out["actual"] - out["forecast"]).abs()
    out["loss_no_change"] = out["y"].abs()
    return out


def smi_block_length(horizon):
    return max(5, int(horizon))


def mae_ratio_with_ci(loss, loss_no_change, block_length, n_boot=2000, seed=20261118):
    """MAE ratio vs. no-change with a paired moving-block bootstrap interval.

    ``ratio = sum(loss) / sum(loss_no_change)``. The interval is the bootstrap
    interval of the mean loss difference, rescaled by the mean no-change loss,
    so ``ratio_high < 1`` exactly when the difference's interval is below 0.
    """
    loss = np.asarray(loss, dtype=float)
    loss_no_change = np.asarray(loss_no_change, dtype=float)
    boot = paired_block_bootstrap(loss, loss_no_change, block_length, n_boot=n_boot, seed=seed)
    scale = loss_no_change.mean()
    ratio = loss.sum() / loss_no_change.sum()
    low, high = 1 + boot["ci_low"] / scale, 1 + boot["ci_high"] / scale
    return {
        "mae_ratio": float(ratio),
        "ratio_low": float(low),
        "ratio_high": float(high),
        "meets_smi": bool(ratio <= SMI_RATIO and high < 1),
    }


def cell_table(results, n_boot=2000, seed=20261118):
    """One row per (model, horizon): n, MAE ratio + interval, SMI verdict, MASE."""
    scored = add_return_columns(results)
    rows = []
    for (model, horizon), group in scored.groupby(["model", "horizon"]):
        if model == NO_CHANGE:
            continue
        group = group.sort_values("origin")
        stats = mae_ratio_with_ci(group["loss"], group["loss_no_change"],
                                  smi_block_length(horizon), n_boot=n_boot, seed=seed)
        rows.append({"model": model, "horizon": horizon, "n": len(group), **stats,
                     "mase": group["scaled_absolute_error"].mean()})
    return pd.DataFrame(rows)


def family_loss_differentials(results, cells):
    """T x k frame of ``loss - loss_no_change`` for each (model, horizon) cell, common origins only."""
    scored = add_return_columns(results)
    columns = {}
    for model, horizon in cells:
        group = scored[(scored["model"] == model) & (scored["horizon"] == horizon)]
        if group.empty:
            raise ValueError(f"no rows for cell ({model}, h={horizon}).")
        columns[f"{model} | h={horizon}"] = pd.Series(
            (group["loss"] - group["loss_no_change"]).to_numpy(), index=group["origin"].to_numpy())
    return pd.DataFrame(columns).sort_index().dropna()


def spa_test(loss_differentials, block_size, reps=1000, seed=20261118, studentize=True):
    """Hansen SPA of "no cell beats no-change".

    Each column of ``loss_differentials`` is a cell's ``|y - f| - |y|`` --
    its loss relative to its *own* no-change benchmark, so cells at different
    horizons can share one test. Passing a zero benchmark makes arch's
    differential ``0 - (|y - f| - |y|) = |y| - |y - f|``, the registered
    ``d``: positive when the cell beats no-change. A small p-value is evidence
    that at least one cell beats no-change.

    Hansen (2005) floors the statistic at zero, ``T = max(max_k t_k, 0)``,
    with ``p = P(T* >= T)``. arch 9.0 does not floor, so when every cell
    *loses* its p-value measures how badly they lose rather than reporting
    "no evidence of a winner". With the floor, ``T = 0`` whenever no cell has
    a positive mean ``d``, and then ``p = 1``; otherwise ``T > 0`` and the
    floored p equals arch's (up to ``>`` vs ``>=`` on a continuous
    statistic). The ``p_*`` keys follow Hansen; ``p_*_arch_raw`` keep arch's
    unfloored values for transparency.
    """
    from arch.bootstrap import SPA

    models = loss_differentials.to_numpy(dtype=float)
    benchmark = np.zeros(models.shape[0])
    spa = SPA(benchmark, models, block_size=int(block_size), reps=int(reps),
              bootstrap="stationary", studentize=studentize, seed=seed)
    spa.compute()
    raw = spa.pvalues
    mean_d = -models.mean(axis=0)                # d = |y| - |y - f|, per cell
    statistic_is_zero = bool(np.all(mean_d <= 0))
    floored = {key: 1.0 if statistic_is_zero else float(raw[key])
               for key in ("lower", "consistent", "upper")}
    return {
        "n_origins": int(models.shape[0]),
        "n_cells": int(models.shape[1]),
        "block_size": int(block_size),
        "reps": int(reps),
        "best_cell": loss_differentials.columns[int(np.argmax(mean_d))],
        "best_mean_d": float(mean_d.max()),
        "statistic_is_zero": statistic_is_zero,
        "p_consistent": floored["consistent"],
        "p_lower": floored["lower"],
        "p_upper": floored["upper"],
        "p_consistent_arch_raw": float(raw["consistent"]),
        "p_lower_arch_raw": float(raw["lower"]),
        "p_upper_arch_raw": float(raw["upper"]),
    }


def descriptive_table(results):
    """Registered descriptive checks per (model, horizon).

    - ``price_level_corr``: correlation of forecast and realised *price* levels.
    - ``directional_accuracy`` vs. ``up_base_rate`` (what "always up" scores),
      with a one-sided binomial p-value against that base rate. Overlapping
      multi-step targets make the binomial p optimistic; it is descriptive.
    - ``dispersion_ratio``: sd of forecast returns over sd of realised returns.
    """
    scored = add_return_columns(results)
    rows = []
    for (model, horizon), group in scored.groupby(["model", "horizon"]):
        y, f = group["y"].to_numpy(), group["f"].to_numpy()
        row = {
            "model": model,
            "horizon": horizon,
            "n": len(group),
            "price_level_corr": float(np.corrcoef(np.exp(group["forecast"]), np.exp(group["actual"]))[0, 1]),
            "up_base_rate": float(np.mean(y > 0)),
            "dispersion_ratio": float(np.std(f) / np.std(y)),
        }
        if model == NO_CHANGE:
            row.update({"directional_accuracy": np.nan, "binom_p_vs_base_rate": np.nan})
        else:
            hits = int(np.sum(np.sign(f) == np.sign(y)))
            row["directional_accuracy"] = hits / len(group)
            row["binom_p_vs_base_rate"] = float(
                binomtest(hits, len(group), p=min(max(row["up_base_rate"], 1e-9), 1 - 1e-9),
                          alternative="greater").pvalue)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Detection limits: planted sign persistence
# ---------------------------------------------------------------------------

def planted_sign_series(abs_returns, delta, rng, start="2000-01-03"):
    """Log-price series with real magnitudes and Markov signs.

    Each sign repeats the previous one with probability ``0.5 + delta``, so
    ``E[s_{t+k} | s_t] = (2 delta)^k s_t``. Magnitudes are ``abs_returns``
    unchanged, keeping the real volatility pattern.
    """
    abs_returns = np.abs(np.asarray(abs_returns, dtype=float))
    n = len(abs_returns)
    stay = rng.random(n - 1) < 0.5 + delta
    signs = np.empty(n)
    signs[0] = rng.choice([-1.0, 1.0])
    for k in range(1, n):
        signs[k] = signs[k - 1] if stay[k - 1] else -signs[k - 1]
    logp = np.concatenate([[0.0], np.cumsum(signs * abs_returns)])
    return pd.Series(logp, index=pd.bdate_range(start, periods=n + 1))


def oracle_forecast(train, horizon, delta, magnitude_window=512):
    """Best forecast under the planted process, knowing ``delta``.

    ``f_h = sum_{k=1..h} (2 delta)^k * s_t * m``, where ``s_t`` is the sign of
    the last return and ``m`` the mean absolute return over the trailing
    window.
    """
    r = np.diff(_values(train))
    s_t = np.sign(r[-1])
    m = np.mean(np.abs(r[-magnitude_window:]))
    steps = (2 * delta) ** np.arange(1, horizon + 1) * s_t * m
    return _values(train)[-1] + np.cumsum(steps)


def wilson_interval(successes, trials, alpha=0.05):
    from statsmodels.stats.proportion import proportion_confint

    low, high = proportion_confint(successes, trials, alpha=alpha, method="wilson")
    return float(low), float(high)
