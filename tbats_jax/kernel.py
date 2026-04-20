"""TBATS innovations-form recursion + likelihood.

Mirrors src/calcTBATS.cpp from forecast::tbats:
    y_hat_t = w @ x_{t-1}
    e_t     = y_t - y_hat_t
    x_t     = F @ x_{t-1} + g * e_t
"""

import jax
import jax.numpy as jnp
from jax import lax

from tbats_jax.spec import TBATSSpec
from tbats_jax.params import unpack
from tbats_jax.matrices import build_matrices
from tbats_jax.admissibility import admissibility_penalty
from tbats_jax.boxcox import boxcox, boxcox_log_jacobian


def tbats_scan(y, x0, F, g, w):
    """Fast path: no NaN handling. For missing-data series call
    `tbats_scan_masked`. Returns (residuals[T], final_state[d])."""

    def step(x_prev, y_t):
        y_hat = jnp.dot(w, x_prev)
        e = y_t - y_hat
        x_t = F @ x_prev + g * e
        return x_t, e

    x_final, residuals = lax.scan(step, x0, y)
    return residuals, x_final


def tbats_scan_masked(y, x0, F, g, w):
    """Missing-data path: when y_t is NaN, skip the innovation update. The
    emitted residual is 0, so it contributes nothing to SSR. Used only when
    the caller has detected NaN values — keeps the non-missing fast path
    bitwise identical to the pre-missing-data implementation.
    """

    def step(x_prev, y_t):
        y_hat = jnp.dot(w, x_prev)
        is_missing = jnp.isnan(y_t)
        e = jnp.where(is_missing, 0.0, y_t - y_hat)
        x_t = F @ x_prev + g * e
        return x_t, e

    x_final, residuals = lax.scan(step, x0, y)
    return residuals, x_final


def _transformed_y(y, params, spec: TBATSSpec, has_missing: bool):
    """Return y_eff for the scan. If Box-Cox is off, just return y. If on,
    transform finite entries (keeping NaN for the mask path when applicable).
    """
    if not spec.use_box_cox:
        return y
    lam = params.box_cox_lambda
    if not has_missing:
        return boxcox(y, lam)
    finite = ~jnp.isnan(y)
    safe_y = jnp.where(finite, y, 1.0)
    z = boxcox(safe_y, lam)
    return jnp.where(finite, z, jnp.nan)


def _scan_for(has_missing: bool):
    return tbats_scan_masked if has_missing else tbats_scan


def sse(y, theta, spec: TBATSSpec, has_missing: bool = False):
    params = unpack(theta, spec)
    F, g, w = build_matrices(spec, params)
    y_eff = _transformed_y(y, params, spec, has_missing)
    residuals, _ = _scan_for(has_missing)(y_eff, params.x0, F, g, w)
    return jnp.sum(residuals ** 2)


def neg_log_likelihood(y, theta, spec: TBATSSpec, has_missing: bool = False):
    """Gaussian negative log-likelihood (up to additive constant).

    Matches forecast::tbats fitTBATS.R line 711-713:
      log.likelihood = n_obs * log(sum(e^2)) - 2 * (lambda - 1) * sum(log(y))
    When `has_missing=True`, NaN observations are excluded from both `n_obs`
    and the Jacobian log-sum.
    """
    params = unpack(theta, spec)
    F, g, w = build_matrices(spec, params)
    y_eff = _transformed_y(y, params, spec, has_missing)
    residuals, _ = _scan_for(has_missing)(y_eff, params.x0, F, g, w)
    if has_missing:
        n_obs = jnp.sum((~jnp.isnan(y)).astype(residuals.dtype))
    else:
        n_obs = jnp.asarray(y.shape[0], dtype=residuals.dtype)
    ssr = jnp.sum(residuals ** 2)
    nll = n_obs * jnp.log(ssr / n_obs)
    if spec.use_box_cox:
        nll = nll - 2.0 * boxcox_log_jacobian(y, params.box_cox_lambda)
    return nll


def penalized_objective(y, theta, spec: TBATSSpec,
                        admissibility_weight: float = 1.0,
                        admissibility_margin: float = 1e-4,
                        gamma_ridge: float = 1e6,
                        has_missing: bool = False):
    """NLL + admissibility barrier + L2 ridge on seasonal gammas.

    `gamma_ridge` mirrors R's implicit gamma regularization (its parscale=1e-5
    causes Nelder-Mead to take tiny gamma steps, effectively keeping ||gamma||
    tiny). Explicit L2 at ~1e6 matches R's behavior and closes the OOS gap on
    long real series (see bench_real.py / taylor).
    """
    params = unpack(theta, spec)
    F, g, w = build_matrices(spec, params)
    y_eff = _transformed_y(y, params, spec, has_missing)
    residuals, _ = _scan_for(has_missing)(y_eff, params.x0, F, g, w)
    if has_missing:
        n_obs = jnp.sum((~jnp.isnan(y)).astype(residuals.dtype))
    else:
        n_obs = jnp.asarray(y.shape[0], dtype=residuals.dtype)
    ssr = jnp.sum(residuals ** 2)
    nll = n_obs * jnp.log(ssr / n_obs)
    if spec.use_box_cox:
        nll = nll - 2.0 * boxcox_log_jacobian(y, params.box_cox_lambda)
    penalty = admissibility_penalty(F, g, w,
                                    margin=admissibility_margin,
                                    weight=admissibility_weight)
    ridge = gamma_ridge * (jnp.sum(params.gamma1 ** 2) + jnp.sum(params.gamma2 ** 2))
    return nll + penalty + ridge


def make_objective(spec: TBATSSpec, y,
                   admissibility_weight: float = 1.0,
                   admissibility_margin: float = 1e-4,
                   gamma_ridge: float = 1e6):
    """Return a JIT-compiled (objective, gradient) pair bound to y + spec."""
    y = jnp.asarray(y)

    def obj(theta):
        return penalized_objective(y, theta, spec,
                                   admissibility_weight=admissibility_weight,
                                   admissibility_margin=admissibility_margin,
                                   gamma_ridge=gamma_ridge)

    jit_obj = jax.jit(obj)
    jit_grad = jax.jit(jax.grad(obj))
    return jit_obj, jit_grad
