"""Online correction of a frozen forecaster under delayed labels (Post 04).

A frozen base model (Chronos-2 zero-shot here) issues a forecast at every
time step. A small corrector sits on top of it and learns from the base
model's past errors -- but the error of an h-step-ahead forecast only exists
h steps after the forecast was issued. This module replays a cached stream
of base forecasts and enforces that rule: the label of origin ``j`` at
horizon ``h`` is handed to a corrector only at stream step ``j + lag``,
with ``lag >= h`` for every causal setting.

Three correctors, each a small analogue rather than a reproduction:

- ``RidgeResidualAdapter`` -- ORCA (Dai et al., NeurIPS 2026,
  arXiv:2606.14222): a linear map from the "context of errors" (recent
  inputs + the base forecast) to the base model's residual, refit from a
  recency-weighted replay buffer. ORCA trains its linear adapter with AdamW;
  closed-form ridge fits the same linear hypothesis class.
- ``RLSBank`` -- Training on the Future (Fentazi, Ameur & Ksentini 2026,
  arXiv:2610.12232): six recursive-least-squares filters at different
  forgetting factors plus a passthrough, combined by a median. Re-implemented
  from the paper's Appendix D pseudocode.
- ``OnlineResidualMemory`` -- Post 01 Demo 2's nearest-neighbor residual
  retrieval, run online: it may only retrieve residuals whose labels have
  already been released.

``BoltzmannRouter`` wraps any corrector with ORCA's error-gated blend back
toward the frozen forecast.
"""

import time
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from src.residual_memory import _normalize


# ---------------------------------------------------------------------------
# Forecast stream
# ---------------------------------------------------------------------------

def base_path_forecast(train, horizon, base_fn, base_kwargs=None):
    """Wrap a base model so ``rolling_origin_evaluate`` keeps its full path.

    The harness only records the horizons being scored; the ORCA-style
    adapter also conditions on the base forecast at every lead time up to
    ``h``, so the full path is passed through as an extra column.
    """
    output = base_fn(train, horizon=horizon, **(base_kwargs or {}))
    forecast = np.asarray(output["forecast"] if isinstance(output, dict) else output, dtype=float)
    return {"forecast": forecast, "base_path": forecast}


@dataclass
class ForecastStream:
    """Per-origin arrays an online corrector replays, one row per stream step.

    ``actual`` is ``None`` on the copy handed to correctors (see
    ``without_targets``) -- correctors only ever see a target through
    ``update``, at its release step.
    """

    origins: pd.Index           # origin timestamps (last observed point), one per step
    horizons: list              # scored horizons, in column order
    base: np.ndarray            # (T, K) base forecast at each scored horizon
    base_path: np.ndarray       # (T, max_horizon) full base forecast path
    context: np.ndarray         # (T, context_length) observations up to each origin
    actual: np.ndarray | None   # (T, K) realized values
    level: float                # pre-stream mean, for standardizing
    scale: float                # pre-stream std, for standardizing

    def without_targets(self):
        return replace(self, actual=None)


def build_stream(series, base_results, context_length):
    """Assemble a ``ForecastStream`` from ``rolling_origin_evaluate`` output.

    ``base_results`` must come from a stride-1 run of ``base_path_forecast``
    so consecutive rows are consecutive time steps -- the release rule
    counts delay in stream steps.
    """
    values = series.to_numpy(dtype=float)
    results = base_results.sort_values(["train_size", "horizon"])
    horizons = sorted(int(h) for h in results["horizon"].unique())
    per_origin = results.groupby("train_size", sort=True)

    positions = np.array(list(per_origin.groups.keys()))
    if len(positions) > 1 and not np.all(np.diff(positions) == 1):
        raise ValueError("base_results must come from a stride-1 rolling-origin run.")
    if positions[0] < context_length:
        raise ValueError("Not enough history before the first origin for context_length.")

    first = per_origin.head(1).set_index("train_size").loc[positions]
    base_path = np.stack(first["base_path"].to_numpy())
    actual = results.pivot(index="train_size", columns="horizon", values="actual").loc[positions, horizons]
    base = results.pivot(index="train_size", columns="horizon", values="forecast").loc[positions, horizons]
    context = np.stack([values[p - context_length:p] for p in positions])

    history = values[:positions[0]]
    return ForecastStream(
        origins=pd.Index(first["origin"].to_numpy()),
        horizons=horizons,
        base=base.to_numpy(dtype=float),
        base_path=base_path,
        context=context,
        actual=actual.to_numpy(dtype=float),
        level=float(history.mean()),
        scale=float(history.std()),
    )


# ---------------------------------------------------------------------------
# Delay-aware replay
# ---------------------------------------------------------------------------

def release_lag(horizon, delay):
    """Stream steps between issuing a forecast and releasing its label.

    ``delay`` is a multiple of the horizon: 1.0 is the minimum causal delay
    (the label is used as soon as it has been observed), 2.0 adds a further
    ``h`` steps of pipeline latency. ``"leaky"`` releases every label one
    step after issue -- before an h > 1 target exists. It is the legacy
    protocol Training on the Future audits, kept only as a reference.
    """
    if delay == "leaky":
        return 1
    lag = int(round(float(delay) * horizon))
    if lag < horizon:
        raise ValueError("A causal delay must be at least 1.0x the horizon; use 'leaky' for the reference.")
    return lag


def run_online(stream, make_corrector, delay=1.0):
    """Replay ``stream`` through one corrector per scored horizon.

    At each step ``t`` every corrector first receives any label released at
    ``t`` (origin ``t - lag``), then issues its forecast for origin ``t``.

    Returns
    -------
    forecasts : numpy.ndarray, shape (T, K)
    elapsed : numpy.ndarray, shape (T,)
        Corrector wall time per step (update + predict, all horizons).
    """
    if stream.actual is None:
        raise ValueError("run_online needs the stream with targets; it hides them from correctors itself.")

    hidden = stream.without_targets()
    n_steps, n_horizons = stream.base.shape
    lags = [release_lag(h, delay) for h in stream.horizons]
    correctors = [make_corrector(hidden, k) for k in range(n_horizons)]

    forecasts = np.empty((n_steps, n_horizons))
    elapsed = np.empty(n_steps)
    for t in range(n_steps):
        start = time.perf_counter()
        for k, corrector in enumerate(correctors):
            j = t - lags[k]
            if j >= 0:
                corrector.update(j, stream.actual[j, k])
            forecasts[t, k] = corrector.predict(t)
        elapsed[t] = time.perf_counter() - start

    return forecasts, elapsed


def corrected_results(base_results, stream, forecasts, elapsed, model_name):
    """Results frame in the harness's schema, for ``summarize`` and plotting.

    ``elapsed_seconds`` is the base model's own per-origin cost plus the
    corrector's per-step cost -- a corrector is never free to run.
    """
    rows = base_results.sort_values(["train_size", "horizon"]).copy()
    rows = rows.drop(columns=["base_path"], errors="ignore")
    n_horizons = len(stream.horizons)
    rows["forecast"] = forecasts.reshape(-1)
    rows["elapsed_seconds"] = rows["elapsed_seconds"].to_numpy() + np.repeat(elapsed, n_horizons)
    rows["model"] = model_name
    rows["scaled_absolute_error"] = (rows["actual"] - rows["forecast"]).abs() / rows["mase_scale"]
    rows["direction_correct"] = (
        np.sign(rows["actual"] - rows["origin_actual"]) == np.sign(rows["forecast"] - rows["origin_actual"])
    )
    return rows.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Correctors
# ---------------------------------------------------------------------------

class RidgeResidualAdapter:
    """ORCA-style residual adapter, fit in closed form.

    Features: the context window and the base path up to ``h``, both
    centered on the context mean and divided by the stream scale, plus an
    intercept. Target: the base model's residual at ``h``, in the same units.
    Refit every ``refit_every`` released labels from the most recent
    ``buffer_size`` samples, weighted by recency with ``half_life`` (in
    labels; ``None`` for equal weights) -- ORCA's decay-sampled FIFO buffer.
    Until ``min_samples`` labels have arrived it passes the base forecast
    through unchanged.
    """

    def __init__(self, stream, k, ridge_alpha=10.0, half_life=None, buffer_size=3000,
                 refit_every=None, min_samples=None):
        self.stream = stream
        self.k = k
        self.h = stream.horizons[k]
        self.ridge_alpha = ridge_alpha
        self.half_life = half_life
        self.buffer_size = buffer_size
        self.refit_every = refit_every or self.h
        n_features = stream.context.shape[1] + self.h + 1
        self.min_samples = min_samples or n_features
        self.X, self.y = [], []
        self.coef = None
        self.since_fit = 0

    def _features(self, i):
        context = self.stream.context[i]
        center = context.mean()
        return np.concatenate([
            (context - center) / self.stream.scale,
            (self.stream.base_path[i, :self.h] - center) / self.stream.scale,
            [1.0],
        ])

    def _fit(self):
        X = np.asarray(self.X[-self.buffer_size:])
        y = np.asarray(self.y[-self.buffer_size:])
        if self.half_life:
            age = np.arange(len(y))[::-1]
            w = 0.5 ** (age / self.half_life)
        else:
            w = np.ones(len(y))
        penalty = self.ridge_alpha * np.eye(X.shape[1])
        penalty[-1, -1] = 0.0  # don't shrink the intercept
        Xw = X * w[:, None]
        self.coef = np.linalg.solve(X.T @ Xw + penalty, Xw.T @ y)

    def update(self, j, actual):
        self.X.append(self._features(j))
        self.y.append((actual - self.stream.base[j, self.k]) / self.stream.scale)
        self.since_fit += 1
        if len(self.y) >= self.min_samples and (self.coef is None or self.since_fit >= self.refit_every):
            self._fit()
            self.since_fit = 0

    def predict(self, t):
        base = self.stream.base[t, self.k]
        if self.coef is None:
            return base
        return base + self.stream.scale * float(self._features(t) @ self.coef)


class RLSBank:
    """Median of six forgetting-factor RLS filters plus a passthrough.

    Per Training on the Future, Section 4 and Appendix D. Features, all in
    standardized units: the base forecast, an intercept, and the last
    observation and seasonal-naive value, each minus the trailing
    ``level_window`` mean. With ``season_length=1`` the seasonal-naive value
    *is* the last observation, so that duplicate feature is dropped.
    """

    FORGETTING_FACTORS = (1.0, 0.9999, 0.9995, 0.999, 0.995, 0.99)

    def __init__(self, stream, k, season_length, level_window, rho=10.0):
        self.stream = stream
        self.k = k
        self.h = stream.horizons[k]
        self.season_length = season_length
        self.level_window = level_window
        self.rho = rho
        n_features = 4 if season_length > 1 else 3
        self.w = np.tile(np.eye(1, n_features)[0], (len(self.FORGETTING_FACTORS), 1))  # passthrough start
        self.P = np.stack([rho * np.eye(n_features) for _ in self.FORGETTING_FACTORS])

    def _z(self, value):
        return (value - self.stream.level) / self.stream.scale

    def _features(self, i):
        context = self.stream.context[i]
        level = context[-self.level_window:].mean()
        features = [self._z(self.stream.base[i, self.k]), 1.0, (context[-1] - level) / self.stream.scale]
        if self.season_length > 1:
            seasonal_naive = context[-self.season_length + (self.h - 1) % self.season_length]
            features.append((seasonal_naive - level) / self.stream.scale)
        return np.asarray(features)

    def update(self, j, actual):
        x, y = self._features(j), self._z(actual)
        n_features = len(x)
        for e, lam in enumerate(self.FORGETTING_FACTORS):
            P = self.P[e]
            Px = P @ x
            gain = Px / (lam + x @ Px)
            self.w[e] += gain * (y - self.w[e] @ x)
            P = (P - np.outer(gain, Px)) / lam
            P = 0.5 * (P + P.T)
            trace_ratio = np.trace(P) / n_features
            if trace_ratio > self.rho:
                P *= self.rho / trace_ratio
            self.P[e] = P

    def predict(self, t):
        x = self._features(t)
        experts = np.append(self.w @ x, x[0])  # last entry: the frozen passthrough
        return self.stream.level + self.stream.scale * float(np.median(experts))


class OnlineResidualMemory:
    """Post 01 Demo 2's residual retrieval, restricted to released labels.

    Memory holds (normalized context window, base residual at ``h``) for
    every origin whose label has been released. At each step, the ``k_neighbors``
    most similar windows vote on the correction, inverse-distance weighted,
    scaled by ``scale_factor`` (fit on validation, as in Post 01).
    """

    def __init__(self, stream, k, lookback, k_neighbors=5, scale_factor=1.0, eps=1e-8):
        self.stream = stream
        self.k = k
        self.lookback = lookback
        self.k_neighbors = k_neighbors
        self.scale_factor = scale_factor
        self.eps = eps
        self.windows, self.residuals = [], []

    def _window(self, i):
        return _normalize(self.stream.context[i, -self.lookback:])

    def update(self, j, actual):
        self.windows.append(self._window(j))
        self.residuals.append(actual - self.stream.base[j, self.k])

    def predict(self, t):
        base = self.stream.base[t, self.k]
        if len(self.residuals) < self.k_neighbors:
            return base
        distances = np.linalg.norm(np.asarray(self.windows) - self._window(t), axis=1)
        nearest = np.argpartition(distances, self.k_neighbors - 1)[:self.k_neighbors]
        inverse = 1.0 / (distances[nearest] + self.eps)
        correction = (inverse / inverse.sum()) @ np.asarray(self.residuals)[nearest]
        return base + self.scale_factor * correction


class BoltzmannRouter:
    """ORCA's error-gated blend between the frozen forecast and a corrector.

    Tracks an EMA of each side's absolute error (standardized) on released
    labels only, comparing the corrector's forecast *as issued* at that
    origin, and weights the corrector by
    ``exp(-e_corr/tau) / (exp(-e_base/tau) + exp(-e_corr/tau))``.
    Starts at an even 0.5 blend before any label arrives.
    """

    def __init__(self, inner, tau=0.1, ema_alpha=0.2):
        self.inner = inner
        self.stream = inner.stream
        self.k = inner.k
        self.tau = tau
        self.ema_alpha = ema_alpha
        self.issued = {}
        self.err_base = self.err_corr = None

    def confidence(self):
        if self.err_base is None:
            return 0.5
        a, b = -self.err_corr / self.tau, -self.err_base / self.tau
        return float(1.0 / (1.0 + np.exp(b - a)))

    def update(self, j, actual):
        self.inner.update(j, actual)
        corrected = self.issued.pop(j)
        e_base = abs(actual - self.stream.base[j, self.k]) / self.stream.scale
        e_corr = abs(actual - corrected) / self.stream.scale
        if self.err_base is None:
            self.err_base, self.err_corr = e_base, e_corr
        else:
            self.err_base = self.ema_alpha * e_base + (1 - self.ema_alpha) * self.err_base
            self.err_corr = self.ema_alpha * e_corr + (1 - self.ema_alpha) * self.err_corr

    def predict(self, t):
        base = self.stream.base[t, self.k]
        corrected = self.inner.predict(t)
        self.issued[t] = corrected
        c = self.confidence()
        return (1 - c) * base + c * corrected
