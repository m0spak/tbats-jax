"""Outer fit: scipy L-BFGS-B driven by jax.grad."""

import time
from dataclasses import dataclass
from typing import Optional, List, Tuple

import jax.numpy as jnp
import numpy as np
# NOTE: scipy is imported lazily inside `fit()` so that merely importing
# this module does not load scipy. Colab's pre-installed scipy is often
# ABI-mismatched against its numpy, and we want the JAX-native path
# (`fit_jax`) to work regardless. Only callers that actually use the
# scipy-backed `fit()` path pay the cost.

from tbats_jax.spec import TBATSSpec
from tbats_jax.params import init_theta
from tbats_jax.kernel import make_objective, neg_log_likelihood
from tbats_jax.seed_state import with_ols_x0, warmup_then_ols


def default_bounds(spec: TBATSSpec) -> List[Tuple[Optional[float], Optional[float]]]:
    """Box bounds in the order of the parameter vector.

    Smoothing coefficients constrained to keep the innovations recursion stable;
    seed state x0 left unbounded.
    """
    bounds: List[Tuple[Optional[float], Optional[float]]] = []
    bounds.append((1e-5, 1.0))             # alpha
    if spec.use_damping:
        bounds.append((0.8, 1.0))           # phi
    if spec.use_trend:
        bounds.append((0.0, 1.0))           # beta
    bounds.extend([(-0.5, 0.5)] * spec.n_gamma)  # gamma1
    bounds.extend([(-0.5, 0.5)] * spec.n_gamma)  # gamma2
    bounds.extend([(None, None)] * spec.state_dim)  # x0 unbounded
    return bounds


@dataclass
class FitResult:
    theta: np.ndarray
    neg_log_lik: float         # penalized objective value reported by scipy
    neg_log_lik_clean: float   # pure Gaussian NLL at theta (no penalty)
    penalty: float             # admissibility penalty at theta
    n_iter: int
    n_feval: int
    n_geval: int
    wall_time: float
    compile_time: float
    converged: bool
    message: str
    spec: TBATSSpec


def fit(
    y,
    spec: TBATSSpec,
    theta0: Optional[np.ndarray] = None,
    method: str = "L-BFGS-B",
    maxiter: int = 500,
    verbose: bool = False,
    bounds: Optional[List[Tuple[Optional[float], Optional[float]]]] = None,
    ols_seed_state: bool = False,
    admissibility_weight: float = 1.0,
    admissibility_margin: float = 1e-4,
    gamma_ridge: float = 1e6,
) -> FitResult:
    """Fit TBATS via scipy L-BFGS-B + jax.grad.

    Regularizer defaults match `fit_jax` so either path produces quality-
    comparable fits at the same knobs.

    ols_seed_state: if True, initialize x0 from the OLS-on-residuals warmup
      that forecast::tbats uses. Currently interacts badly with scipy's
      L-BFGS-B ftol (early termination when x0 gradient is already near zero),
      so it defaults to False. See seed_state.py for the implementation.
    """
    from scipy.optimize import minimize  # lazy: see module-level note

    y_np = np.asarray(y, dtype=np.float64)
    if theta0 is None:
        theta0 = init_theta(spec, y_np)
        if ols_seed_state:
            theta0 = warmup_then_ols(theta0, y_np, spec, warmup_iters=20)
    if bounds is None:
        bounds = default_bounds(spec)

    obj, grad = make_objective(
        spec, y_np,
        admissibility_weight=admissibility_weight,
        admissibility_margin=admissibility_margin,
        gamma_ridge=gamma_ridge,
    )

    # Warm up JIT: first call compiles. Measured separately from fit time.
    t0 = time.perf_counter()
    _ = float(obj(jnp.asarray(theta0)))
    _ = np.asarray(grad(jnp.asarray(theta0)))
    compile_time = time.perf_counter() - t0

    def fun(theta_np):
        return float(obj(jnp.asarray(theta_np)))

    def jac(theta_np):
        return np.asarray(grad(jnp.asarray(theta_np)), dtype=np.float64)

    n_feval = [0]
    n_geval = [0]

    def fun_counted(t):
        n_feval[0] += 1
        return fun(t)

    def jac_counted(t):
        n_geval[0] += 1
        return jac(t)

    # OLS init zeros the x0 gradient block, so scipy's default ftol
    # (≈2.22e-9) can trip after 1 step before smoothing params settle.
    # Tighter tolerances let L-BFGS converge on the smoothing subspace.
    opts = {"maxiter": maxiter}
    if method == "L-BFGS-B":
        opts.update({"ftol": 1e-12, "gtol": 1e-8})

    t0 = time.perf_counter()
    res = minimize(
        fun=fun_counted,
        jac=jac_counted,
        x0=theta0,
        method=method,
        bounds=bounds if method in ("L-BFGS-B", "TNC", "SLSQP") else None,
        options=opts,
    )
    wall_time = time.perf_counter() - t0

    theta_final = np.asarray(res.x)
    theta_j = jnp.asarray(theta_final)
    nll_clean = float(neg_log_likelihood(jnp.asarray(y_np), theta_j, spec))
    pen_val = float(res.fun) - nll_clean

    return FitResult(
        theta=theta_final,
        neg_log_lik=float(res.fun),
        neg_log_lik_clean=nll_clean,
        penalty=pen_val,
        n_iter=int(res.nit),
        n_feval=n_feval[0],
        n_geval=n_geval[0],
        wall_time=wall_time,
        compile_time=compile_time,
        converged=bool(res.success),
        message=str(res.message),
        spec=spec,
    )
