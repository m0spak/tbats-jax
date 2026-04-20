"""End-to-end smoke tests. Run: pytest tests/"""

import numpy as np
import jax
import jax.numpy as jnp

from tbats_jax import TBATSSpec, fit, forecast
from tbats_jax.kernel import neg_log_likelihood, tbats_scan
from tbats_jax.matrices import build_matrices
from tbats_jax.params import init_theta, unpack


def _small_series(n=500):
    rng = np.random.default_rng(0)
    t = np.arange(n)
    y = 5.0 + 0.01 * t + 2.0 * np.sin(2 * np.pi * t / 24.0) + rng.normal(0, 0.3, n)
    return y


def test_spec_dims():
    spec = TBATSSpec(seasonal=((24.0, 3), (168.0, 5)), use_trend=True, use_damping=True)
    assert spec.n_gamma == 8
    assert spec.state_dim == 1 + 1 + 2 * 8  # 18
    assert spec.n_smooth == 1 + 1 + 1 + 2 * 8  # alpha, phi, beta, gamma1, gamma2
    assert spec.n_params == spec.n_smooth + spec.state_dim


def test_scan_matches_reference():
    """Kernel output must match a plain numpy reference implementation."""
    spec = TBATSSpec(seasonal=((24.0, 2),), use_trend=True, use_damping=False)
    y = _small_series(n=200)
    theta = init_theta(spec, y)
    params = unpack(jnp.asarray(theta), spec)
    F, g, w = build_matrices(spec, params)

    resid_jax, _ = tbats_scan(jnp.asarray(y), params.x0, F, g, w)
    resid_jax = np.asarray(resid_jax)

    # Numpy reference
    F_np = np.asarray(F); g_np = np.asarray(g); w_np = np.asarray(w)
    x = np.asarray(params.x0).copy()
    resid_ref = np.zeros(len(y))
    for t, y_t in enumerate(y):
        y_hat = w_np @ x
        e = y_t - y_hat
        x = F_np @ x + g_np * e
        resid_ref[t] = e

    assert np.allclose(resid_jax, resid_ref, atol=1e-6), \
        f"max diff = {np.max(np.abs(resid_jax - resid_ref))}"


def test_fit_converges():
    spec = TBATSSpec(seasonal=((24.0, 2),), use_trend=True, use_damping=False)
    y = _small_series(n=300)
    res = fit(y, spec, maxiter=200)
    assert res.converged or res.n_iter >= 10, res.message
    # Final likelihood should be better than initial
    theta0 = init_theta(spec, y)
    init_nll = float(neg_log_likelihood(jnp.asarray(y), jnp.asarray(theta0), spec))
    assert res.neg_log_lik < init_nll, \
        f"fit did not improve: init={init_nll}, final={res.neg_log_lik}"


def test_grad_finite():
    spec = TBATSSpec(seasonal=((12.0, 2),), use_trend=False, use_damping=False)
    y = _small_series(n=150)
    theta = init_theta(spec, y)
    g = jax.grad(lambda t: neg_log_likelihood(jnp.asarray(y), t, spec))(jnp.asarray(theta))
    g = np.asarray(g)
    assert np.all(np.isfinite(g)), "gradient contains NaN or inf"
    assert np.any(g != 0), "gradient is identically zero"


def test_forecast_shape():
    spec = TBATSSpec(seasonal=((24.0, 2),), use_trend=True, use_damping=False)
    y = _small_series(n=200)
    res = fit(y, spec, maxiter=100)
    h = 48
    preds = forecast(y, jnp.asarray(res.theta), spec, h)
    preds = np.asarray(preds)
    assert preds.shape == (h,)
    assert np.all(np.isfinite(preds))


def test_box_cox_roundtrip():
    """BoxCox then InvBoxCox recovers the input."""
    from tbats_jax.boxcox import boxcox, inv_boxcox
    y = jnp.asarray(np.linspace(1.0, 100.0, 50))
    for lam in (0.1, 0.5, 0.9):
        z = boxcox(y, jnp.asarray(lam))
        back = inv_boxcox(z, jnp.asarray(lam))
        assert np.allclose(np.asarray(back), np.asarray(y), atol=1e-6), \
            f"roundtrip failed at lam={lam}"


def test_auto_fit_converges():
    """auto_fit_jax returns a valid result with k_i <= cap."""
    from tbats_jax import auto_fit_jax
    rng = np.random.default_rng(17)
    n, period = 600, 24
    t = np.arange(n)
    y = 5.0 + 0.005 * t + 2.0 * np.sin(2 * np.pi * t / period) + rng.normal(0, 0.3, n)
    res = auto_fit_jax(y, periods=(float(period),),
                      use_trend=True, use_damping=False,
                      max_steps=200)
    # max_k for m=24 is floor(23/2) = 11
    assert len(res.k_vector) == 1
    assert 1 <= res.k_vector[0] <= 11
    assert np.isfinite(res.aic)
    assert res.n_candidates_tried >= 1


def test_auto_fit_cv_runs():
    """auto_fit_jax_cv returns a valid result — smoke check only."""
    from tbats_jax import auto_fit_jax_cv
    rng = np.random.default_rng(19)
    n, period = 400, 24
    t = np.arange(n)
    y = 5.0 + 0.005 * t + 2.0 * np.sin(2 * np.pi * t / period) + rng.normal(0, 0.3, n)
    res = auto_fit_jax_cv(y, periods=(float(period),),
                          use_trend=True, use_damping=False,
                          val_size=50, max_steps=100)
    assert 1 <= res.k_vector[0] <= 11
    assert np.isfinite(res.val_mae)
    assert res.n_candidates_tried >= 1


def test_heterogeneous_panel():
    """Mix two specs, different lengths; fit_panel_hetero returns per-series theta."""
    from tbats_jax import fit_panel_hetero
    spec_a = TBATSSpec(seasonal=((24.0, 2),), use_trend=True, use_damping=False)
    spec_b = TBATSSpec(seasonal=((12.0, 3),), use_trend=False, use_damping=False)
    rng = np.random.default_rng(11)

    def mk(n, period, seed):
        t = np.arange(n)
        return 3.0 + np.sin(2 * np.pi * t / period) + rng.normal(0, 0.2, n)

    panel = [
        (mk(300, 24, 1), spec_a),
        (mk(400, 12, 2), spec_b),
        (mk(250, 24, 3), spec_a),
        (mk(150, 12, 4), spec_b),
    ]
    out = fit_panel_hetero(panel, max_steps=100)
    assert len(out) == 4
    for r, (y, spec) in zip(out, panel):
        assert r["spec"] is spec
        assert r["theta"].shape == (spec.n_params,)
        assert np.isfinite(r["neg_log_lik_clean"])


def test_missing_data_tolerated():
    """NaN values in y should not corrupt fit or forecast."""
    from tbats_jax import fit_jax
    spec = TBATSSpec(seasonal=((24.0, 2),), use_trend=True, use_damping=False)
    rng = np.random.default_rng(7)
    t = np.arange(600)
    y = 5.0 + 0.005 * t + 2.0 * np.sin(2 * np.pi * t / 24.0) + rng.normal(0, 0.3, 600)
    # Blank out 10% of points randomly
    mask = rng.random(len(y)) < 0.10
    y_masked = y.copy()
    y_masked[mask] = np.nan
    res = fit_jax(y_masked, spec, max_steps=200)
    assert np.isfinite(res.neg_log_lik_clean), \
        f"NLL not finite: {res.neg_log_lik_clean}"
    preds = np.asarray(forecast(y_masked, jnp.asarray(res.theta), spec, 24))
    assert np.all(np.isfinite(preds))


def test_fit_scan_runs():
    """Experimental fit_scan should at least run and return finite output.
    Quality gap to fit_jax is documented in fit_scan.py module docstring —
    this test is a smoke check, not a quality assertion."""
    from tbats_jax import fit_scan
    spec = TBATSSpec(seasonal=((24.0, 2),), use_trend=True, use_damping=False)
    y = _small_series(n=300)
    r = fit_scan(y, spec, max_steps=100)
    assert np.isfinite(r.neg_log_lik_clean)
    preds = np.asarray(forecast(y, jnp.asarray(r.theta), spec, 24))
    assert np.all(np.isfinite(preds))


def test_box_cox_fit_and_forecast():
    """With Box-Cox enabled, fit + forecast stays finite and close to R's scale."""
    from tbats_jax import fit_jax
    spec = TBATSSpec(seasonal=((24.0, 2),), use_trend=True, use_damping=False,
                     use_box_cox=True)
    rng = np.random.default_rng(42)
    t = np.arange(400)
    # positive series with multiplicative seasonality (Box-Cox target)
    y = np.exp(2.0 + 0.001 * t + 0.5 * np.sin(2 * np.pi * t / 24.0)
               + rng.normal(0, 0.1, 400))
    res = fit_jax(y, spec, max_steps=200)
    assert np.isfinite(res.neg_log_lik_clean)
    preds = np.asarray(forecast(y, jnp.asarray(res.theta), spec, 48))
    assert preds.shape == (48,)
    assert np.all(np.isfinite(preds))
    # forecasts should be in the same positive range as y
    assert preds.min() > 0
