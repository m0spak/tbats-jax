"""OLS initialization of the seed state x0.

Mirrors fitTBATS.R lines 327-368:
  1. Run the scan with x0 = 0 under current (F, g, w) to get residuals e.
  2. Build W_tilde where row t = w @ D^t, D = F - g w^T.
     (D governs how perturbations to x0 propagate into each future residual.)
  3. Solve the linear regression e ~ W_tilde @ x0 for x0.

This gives a plausible warmup state that matches what the R reference does,
closing the SSR gap that arises from a naive x0 = [mean(y[:10]), 0, ...].
"""

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

from tbats_jax.spec import TBATSSpec
from tbats_jax.matrices import build_matrices
from tbats_jax.kernel import tbats_scan
from tbats_jax.params import unpack


def _build_W_tilde(F, g, w, n):
    """Stack of row vectors: W[t, :] = w @ D^t for t = 0..n-1."""
    D = F - jnp.outer(g, w)

    def step(w_t, _):
        next_w = w_t @ D
        return next_w, w_t

    _, W = lax.scan(step, w, jnp.zeros(n))
    return W


def ols_seed_state(y: np.ndarray, theta_init: np.ndarray, spec: TBATSSpec) -> np.ndarray:
    """Return the length-`state_dim` OLS estimate of x0.

    `theta_init` need only have correct smoothing params; its x0 slot is
    overwritten on return.
    """
    y_j = jnp.asarray(y)
    theta_j = jnp.asarray(theta_init)
    params = unpack(theta_j, spec)
    F, g, w = build_matrices(spec, params)

    # Residuals under x0 = 0
    zero_x0 = jnp.zeros(spec.state_dim)
    e, _ = tbats_scan(y_j, zero_x0, F, g, w)
    W = _build_W_tilde(F, g, w, y_j.shape[0])

    x0, *_ = jnp.linalg.lstsq(W, e, rcond=None)
    return np.asarray(x0, dtype=np.float64)


def with_ols_x0(theta_init: np.ndarray, y: np.ndarray, spec: TBATSSpec) -> np.ndarray:
    """Overwrite the x0 slot of `theta_init` with the OLS estimate."""
    x0 = ols_seed_state(y, theta_init, spec)
    theta = np.asarray(theta_init, dtype=np.float64).copy()
    theta[-spec.state_dim:] = x0
    return theta


def warmup_then_ols(theta_init: np.ndarray, y: np.ndarray, spec: TBATSSpec,
                    warmup_iters: int = 20) -> np.ndarray:
    """Brief warmup of smoothing params, THEN OLS refit of x0.

    Without this, OLS-fitted x0 can produce a non-admissible F and trap the
    main optimizer in a tight region. The warmup moves smoothing params
    into a sensible regime so OLS sees an already-close-to-admissible model.
    """
    # Deferred import to avoid circularity.
    from scipy.optimize import minimize
    import jax
    import jax.numpy as jnp
    from tbats_jax.kernel import make_objective
    from tbats_jax.fit import default_bounds

    obj, grad = make_objective(spec, y)
    bounds = default_bounds(spec)

    res = minimize(
        fun=lambda t: float(obj(jnp.asarray(t))),
        jac=lambda t: np.asarray(grad(jnp.asarray(t)), dtype=np.float64),
        x0=np.asarray(theta_init, dtype=np.float64),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": warmup_iters, "ftol": 1e-8, "gtol": 1e-6},
    )
    warmed = np.asarray(res.x, dtype=np.float64)
    return with_ols_x0(warmed, y, spec)
