"""Shared rolling-origin evaluation harness.

Every demo notebook uses the same evaluation design so that models are
comparable across the repo: identical forecast origins, identical horizons,
and the same set of metrics (MAE, RMSE, MASE, directional accuracy), plus
runtime.
"""

import time

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Point metrics
# ---------------------------------------------------------------------------

def mae(actual, forecast):
    actual, forecast = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    return np.mean(np.abs(actual - forecast))


def rmse(actual, forecast):
    actual, forecast = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    return np.sqrt(np.mean((actual - forecast) ** 2))


def mase(actual, forecast, training_series, seasonality=1):
    """Mean absolute scaled error, scaled by the in-sample naive/seasonal-naive error."""
    training_series = np.asarray(training_series, dtype=float)
    if len(training_series) <= seasonality:
        raise ValueError("training_series must contain more observations than seasonality.")

    naive_errors = np.abs(training_series[seasonality:] - training_series[:-seasonality])
    scale = np.mean(naive_errors)
    if scale == 0:
        raise ValueError("MASE scale is zero; training_series may be constant.")

    return mae(actual, forecast) / scale


def directional_accuracy(actual, forecast, origin_actual):
    """Fraction of forecasts that got the sign of the change from the origin right."""
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    origin_actual = np.asarray(origin_actual, dtype=float)
    return np.mean(np.sign(actual - origin_actual) == np.sign(forecast - origin_actual))


# ---------------------------------------------------------------------------
# Forecast origins
# ---------------------------------------------------------------------------

def generate_forecast_origins(n_obs, initial_train_size, max_horizon, step=1):
    """Integer positions (end-exclusive train size) usable as forecast origins."""
    if not (0 < initial_train_size < n_obs):
        raise ValueError("initial_train_size must be in (0, n_obs).")
    if max_horizon <= 0 or step <= 0:
        raise ValueError("max_horizon and step must be positive.")

    last_origin = n_obs - max_horizon
    return list(range(initial_train_size, last_origin + 1, step))


# ---------------------------------------------------------------------------
# Rolling-origin evaluation
# ---------------------------------------------------------------------------

def rolling_origin_evaluate(
    series,
    model_fn,
    model_name,
    horizons,
    initial_train_size,
    step=1,
    window="expanding",
    window_size=None,
    mase_seasonality=1,
    model_kwargs=None,
):
    """Run a walk-forward evaluation of one model over one series.

    Parameters
    ----------
    series : pandas.Series
        Full series, DatetimeIndex, ordered.
    model_fn : callable
        ``model_fn(train, horizon, **model_kwargs)`` returning either an
        array of length ``horizon``, or a dict with a ``"forecast"`` key
        plus arbitrary extra diagnostics (e.g. fitted order, residuals)
        that are copied into every result row.
    horizons : list of int
        Horizons to score. The model is always asked to forecast
        ``max(horizons)`` steps; shorter horizons are read off that array.
    window : {"expanding", "rolling"}
        Training-window policy across origins.

    Returns
    -------
    pandas.DataFrame
        One row per (origin, horizon).
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("series must have a DatetimeIndex.")

    model_kwargs = model_kwargs or {}
    horizons = sorted(set(horizons))
    max_horizon = max(horizons)

    origins = generate_forecast_origins(len(series), initial_train_size, max_horizon, step)

    rows = []
    for origin in origins:
        if window == "expanding":
            train = series.iloc[:origin]
        elif window == "rolling":
            if not window_size:
                raise ValueError("window_size is required when window='rolling'.")
            train = series.iloc[max(0, origin - window_size):origin]
        else:
            raise ValueError("window must be 'expanding' or 'rolling'.")

        future = series.iloc[origin:origin + max_horizon]
        origin_date = series.index[origin - 1]
        origin_actual = series.iloc[origin - 1]

        train_values = train.to_numpy(dtype=float)
        mase_scale = np.mean(
            np.abs(train_values[mase_seasonality:] - train_values[:-mase_seasonality])
        )

        start = time.perf_counter()
        output = model_fn(train, horizon=max_horizon, **model_kwargs)
        elapsed = time.perf_counter() - start

        if isinstance(output, dict):
            forecast = np.asarray(output["forecast"], dtype=float)
            extra = {k: v for k, v in output.items() if k != "forecast"}
        else:
            forecast = np.asarray(output, dtype=float)
            extra = {}

        if len(forecast) != max_horizon:
            raise ValueError(f"{model_name} returned {len(forecast)} values; expected {max_horizon}.")

        for horizon in horizons:
            actual_value = future.iloc[horizon - 1]
            forecast_value = forecast[horizon - 1]
            rows.append({
                "model": model_name,
                "origin": origin_date,
                "origin_actual": origin_actual,
                "forecast_date": future.index[horizon - 1],
                "horizon": horizon,
                "actual": actual_value,
                "forecast": forecast_value,
                "mase_scale": mase_scale,
                "scaled_absolute_error": abs(actual_value - forecast_value) / mase_scale,
                "direction_correct": (
                    np.sign(actual_value - origin_actual) == np.sign(forecast_value - origin_actual)
                ),
                "elapsed_seconds": elapsed,
                "train_size": len(train),
                **extra,
            })

    return pd.DataFrame(rows)


def combine_results(results_a, results_b, weight, model_name):
    """Build a synthetic model's results by convex-combining two models' forecasts.

    ``results_a`` and ``results_b`` must come from evaluating the same
    series over the same origins/horizons/window policy (so ``origin_actual``
    and ``mase_scale`` agree row-for-row). ``weight`` is either a single
    float applied at every horizon, or a dict mapping horizon -> weight
    (fit on validation data — see ``src.models.fit_hybrid_weight``).
    """
    key_cols = ["origin", "horizon"]
    left = results_a[key_cols + ["origin_actual", "forecast_date", "actual", "forecast", "mase_scale", "train_size"]]
    right = results_b[key_cols + ["forecast"]]
    merged = left.merge(right, on=key_cols, suffixes=("_a", "_b"))

    if isinstance(weight, dict):
        w = merged["horizon"].map(weight)
        if w.isna().any():
            missing = sorted(merged.loc[w.isna(), "horizon"].unique())
            raise ValueError(f"weight has no entry for horizon(s) {missing}.")
    else:
        w = weight

    merged["forecast"] = w * merged["forecast_a"] + (1 - w) * merged["forecast_b"]
    merged["model"] = model_name
    merged["scaled_absolute_error"] = (merged["actual"] - merged["forecast"]).abs() / merged["mase_scale"]
    merged["direction_correct"] = (
        np.sign(merged["actual"] - merged["origin_actual"]) == np.sign(merged["forecast"] - merged["origin_actual"])
    )
    merged["elapsed_seconds"] = 0.0  # combination itself is ~free; base models' cost is already recorded elsewhere

    return merged.drop(columns=["forecast_a", "forecast_b"])


def summarize(results, group_cols=("model", "horizon"), include_direction=True):
    """Aggregate rolling_origin_evaluate() output into one row per group."""
    group_cols = list(group_cols)
    summaries = []
    for keys, group in results.groupby(group_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, keys))
        row.update({
            "n_forecasts": len(group),
            "mae": mae(group["actual"], group["forecast"]),
            "rmse": rmse(group["actual"], group["forecast"]),
            "mase": group["scaled_absolute_error"].mean(),
            "mean_elapsed_seconds": group["elapsed_seconds"].mean(),
        })
        if include_direction:
            row["directional_accuracy"] = group["direction_correct"].mean()
        summaries.append(row)

    return pd.DataFrame(summaries).sort_values(group_cols).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Paired significance
# ---------------------------------------------------------------------------

def paired_block_bootstrap(loss_a, loss_b, block_length, n_boot=500, ci=0.95, seed=0):
    """Circular moving-block bootstrap interval for ``mean(loss_a - loss_b)``.

    For two models scored on the same, consecutive origins. Overlapping
    multi-step forecasts make per-origin losses autocorrelated, so resampling
    single origins would overstate certainty; resampling blocks of
    ``block_length`` consecutive origins (wrapping around the end) keeps that
    dependence. Negative ``mean_diff`` means model A had lower loss.
    """
    diff = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n = len(diff)
    block_length = max(1, min(int(block_length), n))
    n_blocks = int(np.ceil(n / block_length))

    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block_length)) % n
    boot_means = diff[idx.reshape(n_boot, -1)[:, :n]].mean(axis=1)

    tail = (1 - ci) / 2
    low, high = np.quantile(boot_means, [tail, 1 - tail])
    return {
        "mean_diff": float(diff.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "significant": bool(low > 0 or high < 0),
    }
