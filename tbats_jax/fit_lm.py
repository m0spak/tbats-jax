"""Levenberg-Marquardt fit — scan-based, TPU-compatible.

TBATS is structurally a least-squares problem (minimize SSR of the one-
step-ahead residuals). LM exploits this: each iteration solves a small
normal-equations linear system instead of doing a line search. No
`while_loop` anywhere → static compile graph → TPU-friendly.

Reformulation (key trick):
  Objective = n * log(SSR/n) + gamma_ridge*||gamma||^2 + admissibility

  Reshape as a residual vector r(theta):
    r_scan       = [e_1, e_2, ..., e_T]  (scan residuals, size T)
    r_gamma      = sqrt(gamma_ridge) * [gamma1, gamma2]  (size 2K)
    r_admiss     = sqrt(adm_weight) * max(0, rho(D) - (1-margin))  (size 1)

  Full r has size T + 2K + 1. LM minimizes 0.5 * ||r||^2, which — up
  to the log-SSR nonlinearity — drives toward the same optimum as the
  penalized NLL. The log wrapping doesn't change the minimizer.

  The admissibility is a HINGE (not log-hinge): sum-of-squares only has
  gradient when rho > 1-margin, zero otherwise. This is the concession
  that enables LM — we give up the log barrier's boundary-kissing
  behavior, but in exchange get TPU-viable fixed-iteration convergence.

LM loop (pure scan — no while_loop):
  For each of max_steps:
    r, J     = residuals(theta), jac_forward(residuals)(theta)
    grad     = J.T @ r
    H_gn     = J.T @ J
    delta    = solve(H_gn + lambda*I, -grad)
    theta'   = theta + delta
    ssr_new  = ||r(theta')||^2
    accept   = ssr_new < ssr_best   # scalar jnp.where, no while_loop
    theta    = jnp.where(accept, theta', theta)
    lambda   = jnp.where(accept, lambda/3, lambda*3)
    ssr_best = jnp.where(accept, ssr_new, ssr_best)

Status: experimental but TPU-friendly by construction. Quality comparable
to optimistix on sum-of-squares objectives; may need tuning for very
hard non-convex landscapes.

Cold-start observation: on real-world series with complex local-minima
structure (e.g., forecast::taylor), LM's cold init from `init_theta`
lands in a different local basin than optimistix's BFGS. Warm-starting
from an `fit_jax` solution converges cleanly to within 2%. An optional
`adam_steps` / `adam_lr` knob runs Adam in scan before LM kicks in —
available but didn't close the gap in our Taylor tests. Multi-start or
data-driven init are the real fixes (future work).
"""

import time
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

from tbats_jax.spec import TBATSSpec
from tbats_jax.params import init_theta, unpack
from tbats_jax.matrices import build_matrices
from tbats_jax.kernel import tbats_scan, neg_log_likelihood
from tbats_jax.admissibility import _spectral_radius_eigvals, _spectral_radius_power_iter
from tbats_jax.transforms import raw_to_natural, natural_to_raw

try:
    import optax
    _HAVE_OPTAX = True
except ImportError:
    _HAVE_OPTAX = False


@dataclass
class FitResultLM:
    theta: np.ndarray
    theta_raw: np.ndarray
    neg_log_lik_clean: float    # pure NLL at final theta (no penalty)
    ssr: float
    final_rho: float            # max|eig(D)| at solution
    max_steps: int
    wall_time: float
    compile_time: float
    spec: TBATSSpec


def _residuals(theta_raw, y, spec, gamma_ridge, adm_weight, adm_margin,
               rho_method):
    """Return residual vector whose squared sum equals the LM objective.

    Composition:
      r[:T]              scan innovations e_t
      r[T:T+2K]          sqrt(gamma_ridge) * gammas         (ridge)
      r[T+2K]            sqrt(adm_weight) * hinge_violation  (admissibility)
    """
    theta = raw_to_natural(theta_raw, spec)
    params = unpack(theta, spec)
    F, g, w = build_matrices(spec, params)
    res_scan, _ = tbats_scan(y, params.x0, F, g, w)

    # gamma ridge as residuals
    K = spec.n_gamma
    reg_gamma = jnp.sqrt(gamma_ridge) * jnp.concatenate([params.gamma1, params.gamma2])

    # admissibility hinge as a single residual
    if rho_method == "eigvals":
        rho = _spectral_radius_eigvals(F - jnp.outer(g, w))
    else:
        rho = _spectral_radius_power_iter(F - jnp.outer(g, w), n_iter=10)
    violation = jnp.maximum(rho - (1.0 - adm_margin), 0.0)
    reg_adm = jnp.sqrt(adm_weight) * jnp.atleast_1d(violation)

    return jnp.concatenate([res_scan, reg_gamma, reg_adm])


def _lm_scan(theta0, r_fn, max_steps, lam0=1.0):
    """Fixed-iteration LM. `r_fn` is closed over — only theta, lam, ssr
    travel through the scan carry (all JAX types, no function objects)."""
    r0 = r_fn(theta0)
    ssr0 = jnp.sum(r0 ** 2)
    d = theta0.shape[0]
    I = jnp.eye(d, dtype=theta0.dtype)

    def step(state, _):
        theta, lam, ssr_cur = state
        r = r_fn(theta)
        J = jax.jacfwd(r_fn)(theta)  # (n_resid, n_params)
        grad = J.T @ r
        H = J.T @ J
        delta = jnp.linalg.solve(H + lam * I, -grad)

        theta_trial = theta + delta
        r_trial = r_fn(theta_trial)
        ssr_trial = jnp.sum(r_trial ** 2)

        accept = (ssr_trial < ssr_cur) & jnp.isfinite(ssr_trial)
        theta_new = jnp.where(accept, theta_trial, theta)
        ssr_new = jnp.where(accept, ssr_trial, ssr_cur)
        lam_new = jnp.where(
            accept,
            jnp.maximum(lam / 3.0, 1e-10),
            jnp.minimum(lam * 3.0, 1e12),
        )
        return (theta_new, lam_new, ssr_new), None

    init = (theta0, jnp.asarray(lam0), ssr0)
    (theta_final, _, _), _ = lax.scan(step, init, None, length=max_steps)
    return theta_final


def _adam_scan(theta0, grad_fn, n_steps, lr):
    """Pure-scan Adam warmup. Finds a good basin before LM polish."""
    if not _HAVE_OPTAX:
        raise ImportError("optax is required for two-phase LM (adam_steps>0)")
    tx = optax.adam(learning_rate=lr)
    state = tx.init(theta0)

    def step(carry, _):
        theta, state = carry
        g = grad_fn(theta)
        updates, state = tx.update(g, state, theta)
        theta = optax.apply_updates(theta, updates)
        return (theta, state), None

    (theta_final, _), _ = lax.scan(step, (theta0, state), None, length=n_steps)
    return theta_final


def fit_lm(
    y,
    spec: TBATSSpec,
    theta0: Optional[np.ndarray] = None,
    max_steps: int = 200,
    lam0: float = 1.0,
    gamma_ridge: float = 1e6,
    admissibility_weight: float = 1e4,
    admissibility_margin: float = 1e-3,
    rho_method: str = "auto",
    adam_steps: int = 0,
    adam_lr: float = 1e-2,
) -> FitResultLM:
    """LM fit using a scan-based Marquardt adaptation.

    No `while_loop` anywhere — compiles quickly on any backend including
    TPU. Uses the hinge (not log-hinge) admissibility form so the full
    objective has sum-of-squares structure that GN/LM exploits.

    Defaults: lam0=1.0 starts mildly damped; accept → halve, reject →
    triple (Marquardt's original rule). max_steps=200 is usually plenty
    because LM converges quadratically when close to a minimum.
    """
    if rho_method == "auto":
        rho_method = "eigvals" if jax.default_backend() == "cpu" else "power"

    y_np = np.asarray(y, dtype=np.float64)
    if theta0 is None:
        theta0 = init_theta(spec, y_np)
    theta_raw0 = natural_to_raw(theta0, spec)
    y_j = jnp.asarray(y_np)

    def r_fn(theta_raw):
        return _residuals(
            theta_raw, y_j, spec,
            gamma_ridge=gamma_ridge,
            adm_weight=admissibility_weight,
            adm_margin=admissibility_margin,
            rho_method=rho_method,
        )

    # Objective function for Adam phase: 0.5 * ||r||^2 (same as LM target).
    def loss_fn(theta_raw):
        r = r_fn(theta_raw)
        return 0.5 * jnp.sum(r ** 2)

    grad_fn = jax.grad(loss_fn)

    @jax.jit
    def run(theta_raw_init):
        th = theta_raw_init
        if adam_steps > 0:
            th = _adam_scan(th, grad_fn, n_steps=adam_steps, lr=adam_lr)
        return _lm_scan(th, r_fn, max_steps=max_steps, lam0=lam0)

    t0 = time.perf_counter()
    x = run(jnp.asarray(theta_raw0))
    jax.block_until_ready(x)
    compile_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    x = run(jnp.asarray(theta_raw0))
    jax.block_until_ready(x)
    wall_time = time.perf_counter() - t0

    theta_raw = np.asarray(x)
    theta = np.asarray(raw_to_natural(jnp.asarray(theta_raw), spec))
    nll = float(neg_log_likelihood(jnp.asarray(y_np), jnp.asarray(theta), spec))

    # Report final rho diagnostic.
    params = unpack(jnp.asarray(theta), spec)
    F, g, w = build_matrices(spec, params)
    D = np.asarray(F) - np.outer(np.asarray(g), np.asarray(w))
    final_rho = float(np.max(np.abs(np.linalg.eigvals(D))))

    ssr = float(np.exp(nll / len(y_np)) * len(y_np))
    return FitResultLM(
        theta=theta,
        theta_raw=theta_raw,
        neg_log_lik_clean=nll,
        ssr=ssr,
        final_rho=final_rho,
        max_steps=max_steps,
        wall_time=wall_time,
        compile_time=compile_time,
        spec=spec,
    )


def fit_panel_lm(
    ys: np.ndarray,
    spec: TBATSSpec,
    max_steps: int = 200,
    lam0: float = 1.0,
    gamma_ridge: float = 1e6,
    admissibility_weight: float = 1e4,
    admissibility_margin: float = 1e-3,
    rho_method: str = "auto",
):
    """Batched LM fit across a panel. Returns (theta_raw: (N, d),
    compile_time, wall_time). TPU-compatible by construction.
    """
    if rho_method == "auto":
        rho_method = "eigvals" if jax.default_backend() == "cpu" else "power"

    ys = np.asarray(ys, dtype=np.float64)
    N, T = ys.shape

    theta_raw_stack = np.stack(
        [natural_to_raw(init_theta(spec, ys[i]), spec) for i in range(N)],
        axis=0,
    )
    raw_stack = jnp.asarray(theta_raw_stack)
    y_j = jnp.asarray(ys)

    def make_r_fn(y_i):
        def r_fn(theta_raw):
            return _residuals(
                theta_raw, y_i, spec,
                gamma_ridge=gamma_ridge,
                adm_weight=admissibility_weight,
                adm_margin=admissibility_margin,
                rho_method=rho_method,
            )
        return r_fn

    def fit_one(theta_raw_init, y_i):
        return _lm_scan(theta_raw_init, make_r_fn(y_i),
                        max_steps=max_steps, lam0=lam0)

    batched = jax.jit(jax.vmap(fit_one, in_axes=(0, 0)))

    t0 = time.perf_counter()
    out = batched(raw_stack, y_j)
    jax.block_until_ready(out)
    compile_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    out = batched(raw_stack, y_j)
    jax.block_until_ready(out)
    wall_time = time.perf_counter() - t0

    return np.asarray(out), compile_time, wall_time
