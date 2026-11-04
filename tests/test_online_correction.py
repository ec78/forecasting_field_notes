import numpy as np
import pandas as pd
import pytest

from src.evaluation import paired_block_bootstrap, rolling_origin_evaluate
from src.online_correction import (
    BoltzmannRouter,
    ForecastStream,
    OnlineResidualMemory,
    RidgeResidualAdapter,
    RLSBank,
    base_path_forecast,
    build_stream,
    corrected_results,
    release_lag,
    run_online,
)


def _stream(n=400, horizons=(1, 3), context_length=8, seed=0, actual_fn=None):
    rng = np.random.default_rng(seed)
    max_h = max(horizons)
    base_path = rng.normal(size=(n, max_h))
    base = np.stack([base_path[:, h - 1] for h in horizons], axis=1)
    context = rng.normal(size=(n, context_length))
    actual = actual_fn(base, context) if actual_fn else base + rng.normal(scale=0.1, size=base.shape)
    return ForecastStream(
        origins=pd.Index(np.arange(n)), horizons=list(horizons), base=base, base_path=base_path,
        context=context, actual=actual, level=0.0, scale=1.0,
    )


class _Spy:
    """Records every (step, released origin) pair and whether targets were visible."""

    calls = []

    def __init__(self, stream, k):
        self.stream, self.k, self.t = stream, k, None
        _Spy.saw_targets = stream.actual is not None

    def update(self, j, actual):
        _Spy.calls.append((self.k, self.t, j))

    def predict(self, t):
        self.t = t + 1  # the next update call happens at step t + 1
        return self.stream.base[t, self.k]


# ---------------------------------------------------------------------------
# Release rule
# ---------------------------------------------------------------------------

def test_release_lag_is_causal_by_default_and_rejects_sub_horizon_delays():
    assert release_lag(24, 1.0) == 24
    assert release_lag(24, 2.0) == 48
    assert release_lag(24, "leaky") == 1
    with pytest.raises(ValueError):
        release_lag(24, 0.5)


@pytest.mark.parametrize("delay", [1.0, 2.0])
def test_run_online_never_releases_a_label_before_it_is_observed(delay):
    _Spy.calls = []
    stream = _stream(n=60, horizons=(1, 5))
    run_online(stream, _Spy, delay=delay)

    assert not _Spy.saw_targets
    assert _Spy.calls
    for k, t, j in _Spy.calls:
        h = stream.horizons[k]
        # Origin j's h-step target is observed once step j + h is reached.
        assert j + h <= t
        assert t - j == release_lag(h, delay)


def test_leaky_reference_really_does_release_unobserved_targets():
    _Spy.calls = []
    stream = _stream(n=30, horizons=(5,))
    run_online(stream, _Spy, delay="leaky")
    assert all(t - j == 1 for _, t, j in _Spy.calls)
    assert any(j + 5 > t for _, t, j in _Spy.calls)


# ---------------------------------------------------------------------------
# Correctors
# ---------------------------------------------------------------------------

def test_rls_bank_starts_at_passthrough_and_learns_a_systematic_bias():
    stream = _stream(n=600, horizons=(2,), actual_fn=lambda base, ctx: base + 1.5)
    bank = RLSBank(stream.without_targets(), 0, season_length=4, level_window=4)
    assert bank.predict(0) == pytest.approx(stream.base[0, 0])

    forecasts, _ = run_online(stream, lambda s, k: RLSBank(s, k, season_length=4, level_window=4))
    late = slice(400, None)
    np.testing.assert_allclose(forecasts[late, 0], stream.actual[late, 0], atol=0.05)


def test_ridge_adapter_passes_through_until_min_samples_then_fits_linear_residual():
    def actual_fn(base, ctx):
        return base + 0.5 * (ctx[:, -1] - ctx.mean(axis=1))[:, None]

    stream = _stream(n=500, horizons=(3,), context_length=6, actual_fn=actual_fn)
    make = lambda s, k: RidgeResidualAdapter(s, k, ridge_alpha=1e-6, min_samples=50)
    forecasts, _ = run_online(stream, make)

    # With lag 3, step t has seen origins 0..t-3 (t-2 labels): the 50th arrives at step 52.
    np.testing.assert_allclose(forecasts[:52, 0], stream.base[:52, 0])
    assert forecasts[52, 0] != pytest.approx(stream.base[52, 0])
    np.testing.assert_allclose(forecasts[300:, 0], stream.actual[300:, 0], atol=1e-6)


def test_online_residual_memory_only_remembers_released_labels():
    stream = _stream(n=40, horizons=(4,))
    memory = OnlineResidualMemory(stream.without_targets(), 0, lookback=4, k_neighbors=2)
    run_online(stream, lambda s, k: memory)
    # Labels for origins 0..(40 - 1 - 4) have been released by the last step.
    assert len(memory.residuals) == 40 - 4


class _FixedCorrector:
    def __init__(self, stream, k, offset):
        self.stream, self.k, self.offset = stream, k, offset

    def update(self, j, actual):
        pass

    def predict(self, t):
        return self.stream.base[t, self.k] + self.offset


@pytest.mark.parametrize("offset, expect_trust", [(1.0, True), (-3.0, False)])
def test_router_trusts_a_corrector_that_beats_base_and_drops_one_that_does_not(offset, expect_trust):
    stream = _stream(n=200, horizons=(2,), actual_fn=lambda base, ctx: base + 1.0)
    routers = []

    def make(s, k):
        routers.append(BoltzmannRouter(_FixedCorrector(s, k, offset), tau=0.1))
        return routers[-1]

    run_online(stream, make)
    c = routers[0].confidence()
    assert (c > 0.99) if expect_trust else (c < 0.01)


# ---------------------------------------------------------------------------
# Stream assembly and output schema
# ---------------------------------------------------------------------------

def test_build_stream_aligns_base_path_context_and_targets_with_the_harness():
    index = pd.date_range("2026-01-01", periods=120, freq="h")
    series = pd.Series(np.arange(120, dtype=float), index=index)

    def dummy(train, horizon):
        return np.full(horizon, train.iloc[-1]) + np.arange(1, horizon + 1) * 0.1

    results = rolling_origin_evaluate(
        series, base_path_forecast, "base", [1, 4], initial_train_size=100, step=1,
        model_kwargs={"base_fn": dummy},
    )
    stream = build_stream(series, results, context_length=10)

    assert stream.horizons == [1, 4]
    np.testing.assert_allclose(stream.context[0], np.arange(90, 100))
    np.testing.assert_allclose(stream.base[0], [99.1, 99.4])
    np.testing.assert_allclose(stream.base_path[0, :4], [99.1, 99.2, 99.3, 99.4])
    np.testing.assert_allclose(stream.actual[0], [100.0, 103.0])

    out = corrected_results(results, stream, stream.base + 1.0, np.zeros(len(stream.base)), "shifted")
    first = out[out["train_size"] == 100].sort_values("horizon")
    np.testing.assert_allclose(first["forecast"], [100.1, 100.4])
    assert "base_path" not in out.columns


# ---------------------------------------------------------------------------
# Paired bootstrap
# ---------------------------------------------------------------------------

def test_paired_block_bootstrap_flags_a_real_gap_and_not_noise():
    rng = np.random.default_rng(1)
    noise = rng.normal(size=1000)

    same = paired_block_bootstrap(noise, rng.normal(size=1000), block_length=20)
    assert not same["significant"]
    assert same["ci_low"] < 0 < same["ci_high"]

    gap = paired_block_bootstrap(noise - 0.5, noise, block_length=20)
    assert gap["significant"] and gap["ci_high"] < 0
    assert gap["mean_diff"] == pytest.approx(-0.5)
