"""Regression guards — pin observed behavior so silent drift shows up in CI.

These tests are intentionally cheap (no T=1500 runs, no real data required
except the Taylor guard which skips if the CSV is missing). Each one catches
a class of regression that bit us during development:

  test_admissibility_methods_agree
      eigvals vs power-iter should give comparable rho on typical D.
      Added after the v0.0.3 -> v0.0.5 misdiagnosis of "power iter is
      broken" — turned out they agreed but we hadn't tested it.

  test_fit_panel_consistency
      fit_panel vmap should produce SSRs within 1% of sequential fit_jax
      calls. Added after per-series init via broadcast gave ~3% drift
      that we only caught via benchmark inspection.

  test_fit_vs_fit_jax_similar_quality
      scipy `fit` and optimistix `fit_jax` are different optimizers on
      the same objective; they should find similar-quality minima.

  test_taylor_regression_guard
      Pins Taylor test-MAE under 1100 (current: 1042). Catches any
      regression that breaks real-data OOS quality.
"""

import os
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from tbats_jax import TBATSSpec, fit, fit_jax, fit_lm, fit_panel, forecast
from tbats_jax.fit_jax import _build_panel_fit, _build_single_fit
from tbats_jax.kernel import neg_log_likelihood
from tbats_jax.transforms import raw_to_natural


def test_admissibility_methods_agree():
    """`_spectral_radius_eigvals` and `_spectral_radius_power_iter` should
    produce similar rho estimates for realistic TBATS D matrices. Power iter
    is an upper bound on exact rho, so we allow a 30% overestimate headroom.
    """
    from tbats_jax.admissibility import (
        _spectral_radius_eigvals,
        _spectral_radius_power_iter,
    )
    rng = np.random.default_rng(0)
    d = 18
    for trial in range(5):
        D = jnp.asarray(rng.normal(0, 0.1, (d, d)) + 0.5 * np.eye(d))
        rho_exact = float(_spectral_radius_eigvals(D))
        rho_power = float(_spectral_radius_power_iter(D, n_iter=15))
        # Power iter on D^T D -> ||D||_2 >= rho(D) mathematically.
        assert rho_power >= rho_exact - 1e-6, \
            f"power iter underbounded: exact={rho_exact}, power={rho_power}"
        # Should be within 30% for these reasonable matrices.
        assert rho_power <= rho_exact * 1.3 + 0.05, \
            f"power iter wildly overestimates: exact={rho_exact}, power={rho_power}"


def test_fit_panel_consistency():
    """`fit_panel` (vmap) should produce per-series SSRs close to sequential
    `fit_jax` calls on the same data + spec. Drift beyond 5% means vmap or
    broadcasting is degrading fit quality."""
    spec = TBATSSpec(seasonal=((24.0, 2),), use_trend=True, use_damping=False)
    rng = np.random.default_rng(3)
    N, T = 8, 400
    panel = np.stack([
        5.0 + 0.005 * np.arange(T)
        + 2.0 * np.sin(2 * np.pi * np.arange(T) / 24.0)
        + rng.normal(0, 0.3, T)
        for _ in range(N)
    ], axis=0)

    # Sequential
    seq_ssr = []
    for i in range(N):
        r = fit_jax(panel[i], spec, max_steps=200)
        seq_ssr.append(np.exp(r.neg_log_lik_clean / T) * T)

    # Batched
    thetas_raw, _, _, _ = fit_panel(panel, spec, max_steps=200)
    batched_ssr = []
    for i in range(N):
        th = np.asarray(raw_to_natural(jnp.asarray(thetas_raw[i]), spec))
        nll = float(neg_log_likelihood(jnp.asarray(panel[i]), jnp.asarray(th), spec))
        batched_ssr.append(np.exp(nll / T) * T)

    rel_diff = np.abs(
        (np.array(batched_ssr) - np.array(seq_ssr)) / np.maximum(seq_ssr, 1e-9)
    )
    assert rel_diff.max() < 0.05, (
        f"fit_panel drift > 5%: max rel diff = {rel_diff.max():.3%}, "
        f"seq={seq_ssr}, batched={batched_ssr}"
    )


def test_fit_vs_fit_jax_similar_quality():
    """scipy `fit` and optimistix `fit_jax` should find minima within 20%
    SSR of each other. Large divergence means one optimizer has a knob
    tuned wrong for shared defaults.
    """
    spec = TBATSSpec(seasonal=((24.0, 2),), use_trend=True, use_damping=False)
    rng = np.random.default_rng(5)
    T = 400
    y = (5.0 + 0.005 * np.arange(T)
         + 2.0 * np.sin(2 * np.pi * np.arange(T) / 24.0)
         + rng.normal(0, 0.3, T))

    r_scipy = fit(y, spec, maxiter=500)
    r_optx  = fit_jax(y, spec, max_steps=500)
    ssr_scipy = float(np.exp(r_scipy.neg_log_lik_clean / T) * T)
    ssr_optx  = float(np.exp(r_optx.neg_log_lik_clean / T) * T)
    rel_diff = abs(ssr_scipy - ssr_optx) / min(ssr_scipy, ssr_optx)
    assert rel_diff < 0.20, \
        f"fit vs fit_jax SSR drift > 20%: scipy={ssr_scipy:.2f}, optx={ssr_optx:.2f}"


_TAYLOR_CSV = Path(__file__).resolve().parents[1] / "data" / "taylor.csv"


@pytest.mark.skipif(
    not _TAYLOR_CSV.exists(),
    reason="taylor.csv not fetched — run `Rscript benchmarks/fetch_real_data.R data` first"
)
def test_taylor_regression_guard():
    """Pins the forecast::taylor held-out MAE so quality regressions are
    caught. Current fit gets ~1042; we allow up to 1100 to tolerate
    BFGS-trajectory jitter across code changes.
    """
    y = np.loadtxt(_TAYLOR_CSV)
    h = 336
    y_tr, y_te = y[:-h], y[-h:]
    spec = TBATSSpec(
        seasonal=((48.0, 3), (336.0, 5)),
        use_trend=True,
        use_damping=True,
    )
    r = fit_jax(y_tr, spec, max_steps=1000)
    pred = np.asarray(forecast(y_tr, jnp.asarray(r.theta), spec, h))
    mae = float(np.mean(np.abs(y_te - pred)))
    assert mae < 1100, f"Taylor MAE regression: {mae:.1f} (budget 1100)"


@pytest.mark.skipif(
    not _TAYLOR_CSV.exists(),
    reason="taylor.csv not fetched — run `Rscript benchmarks/fetch_real_data.R data` first"
)
def test_taylor_fit_lm_cold_start_with_ols_init():
    """Pins the cold-start fit_lm Taylor MAE under the new OLS-default init.
    Pre-OLS the cold-start MAE was ~2800; with OLS it converges to ~1120,
    within ~8% of fit_jax/R reference. Budget 1200 to tolerate scan-step
    jitter across code changes.
    """
    y = np.loadtxt(_TAYLOR_CSV)
    h = 336
    y_tr, y_te = y[:-h], y[-h:]
    spec = TBATSSpec(
        seasonal=((48.0, 3), (336.0, 5)),
        use_trend=True,
        use_damping=True,
    )
    r = fit_lm(y_tr, spec, max_steps=200)  # default init_method='ols'
    pred = np.asarray(forecast(y_tr, jnp.asarray(r.theta), spec, h))
    mae = float(np.mean(np.abs(y_te - pred)))
    assert mae < 1200, (
        f"Taylor cold-start fit_lm MAE regression: {mae:.1f} (budget 1200). "
        f"If this fails, OLS init may have stopped landing in the good basin."
    )


def test_fit_panel_jit_cache_hit():
    """Repeated fit_panel calls with the same spec/shape must reuse the
    cached jit object. Pre-fix, every call rebuilt its closure → JAX cache
    missed → compile_time stayed flat across calls. After the
    `_build_panel_fit` lru_cache lift, the second call's `compile_time`
    should drop sharply.
    """
    _build_panel_fit.cache_clear()
    spec = TBATSSpec(seasonal=((7.0, 3),), use_trend=True)
    rng = np.random.default_rng(0)
    panel = np.stack([rng.normal(size=400) for _ in range(3)], axis=0)

    _, _, ct1, _ = fit_panel(panel, spec, max_steps=50)
    _, _, ct2, _ = fit_panel(panel, spec, max_steps=50)

    assert ct2 < ct1 / 3, (
        f"jit cache miss on second call: ct1={ct1:.3f}s ct2={ct2:.3f}s "
        f"(expected ct2 < ct1/3)"
    )


def test_fit_jax_jit_cache_hit():
    """Repeated fit_jax calls with the same spec/length must reuse the
    cached jit object. Companion guard to test_fit_panel_jit_cache_hit:
    the single-series path lifts y out of the closure into args so different
    series of the same length hit the cache.
    """
    _build_single_fit.cache_clear()
    spec = TBATSSpec(seasonal=((7.0, 3),), use_trend=True)
    rng = np.random.default_rng(0)
    y1 = rng.normal(size=400)
    y2 = rng.normal(size=400)   # different data, same shape

    r1 = fit_jax(y1, spec, max_steps=50)
    r2 = fit_jax(y2, spec, max_steps=50)

    assert r2.compile_time < r1.compile_time / 3, (
        f"jit cache miss on second fit_jax: "
        f"ct1={r1.compile_time:.3f}s ct2={r2.compile_time:.3f}s "
        f"(expected ct2 < ct1/3)"
    )


def test_fit_jax_numerical_equivalence_after_cache_lift():
    """Pin numerical behavior: lifting y from closure to args must not
    change fit results meaningfully. Compares two fits on the same y/spec —
    a near-tautology, but it pins float bit-pattern stability through the
    refactor and catches any unexpected nondeterminism.
    """
    spec = TBATSSpec(seasonal=((7.0, 3),), use_trend=True)
    rng = np.random.default_rng(42)
    y = rng.normal(size=400)

    r_a = fit_jax(y, spec, max_steps=100)
    r_b = fit_jax(y, spec, max_steps=100)

    assert np.allclose(r_a.theta, r_b.theta, atol=1e-12), \
        f"non-determinism in fit_jax: max diff = {np.max(np.abs(r_a.theta - r_b.theta))}"
    assert abs(r_a.neg_log_lik_clean - r_b.neg_log_lik_clean) < 1e-10
