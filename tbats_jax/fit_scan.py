"""Fixed-iteration BFGS expressed as `lax.scan` — TPU-friendly.

**Status: EXPERIMENTAL.** On our TBATS objective (non-convex, log-hinge
barrier with non-smooth gradient at the boundary, large initial gradient
magnitude) this hand-rolled BFGS converges to ~25-30% worse SSR than
`optimistix.BFGS` (fit_jax). On Taylor real data the MAE gap is ~5×.
The gap is fundamental to what hand-rolled BFGS can achieve in a
fixed-iteration budget without optimistix's trust-region, Wolfe
conditions, restart logic, etc.

Kept in the tree as a base for future work on TPU-viable fitting. Do
NOT use for production; use `fit_jax` (GPU/CPU) instead.

Why the scan-based form matters anyway:
  `optimistix.BFGS` uses `lax.while_loop` which TPU XLA compiles very
  slowly (24 min on v5e-1 for a workload GPU finishes in 24 s). A
  fixed-length scan compiles in seconds on any backend. Once the
  convergence-quality gap is closed (via e.g. Wolfe line search, trust
  region, or switching to a JAX-native L-BFGS like the one in optax),
  this path unlocks TPU.

Benefits of the scan form (whenever quality is good enough):
  - TPU compile goes from minutes to seconds
  - No vmap lockstep: every series gets the same budget
  - XLA can unroll / fuse across steps aggressively

Current caveats:
  - Plateaus at higher SSR than fit_jax (see above).
  - Line search is Armijo-only with 20-halving budget; no curvature check.
  - H0 pre-scaled by 1/||g0||; no self-scaling restart on stalls.

API mirrors `fit_jax.fit_jax` / `fit_panel` for drop-in-compatible shape.
"""

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

from tbats_jax.spec import TBATSSpec
from tbats_jax.params import init_theta
from tbats_jax.kernel import penalized_objective, neg_log_likelihood
from tbats_jax.transforms import raw_to_natural, natural_to_raw


@dataclass
class FitResultScan:
    theta: np.ndarray
    theta_raw: np.ndarray
    neg_log_lik: float          # final penalized objective
    neg_log_lik_clean: float    # pure NLL (no penalty)
    penalty: float
    max_steps: int
    wall_time: float
    compile_time: float
    spec: TBATSSpec


# --- Core BFGS-as-scan --------------------------------------------------------


def _backtracking_line_search(f, x, p, f_x, g, c1=1e-4, n_halvings=10):
    """Evaluate f at alpha = 1, 1/2, ..., 1/2^(n-1); return the first alpha
    satisfying the Armijo sufficient-decrease condition, or **zero** if none
    do (a non-step is safer than a regressive tiny step — keeps H healthy).
    """
    alphas = 0.5 ** jnp.arange(n_halvings, dtype=f_x.dtype)
    gp = jnp.dot(g, p)  # directional derivative (negative for descent)

    def eval_at(alpha):
        return f(x + alpha * p)

    f_trials = jax.vmap(eval_at)(alphas)
    # Accept only if genuinely decreases (Armijo) AND stays finite.
    accepted = (f_trials <= f_x + c1 * alphas * gp) & jnp.isfinite(f_trials)
    any_accepted = accepted.any()
    first_idx = jnp.argmax(accepted.astype(jnp.int32))
    alpha_chosen = jnp.where(any_accepted, alphas[first_idx], 0.0)
    f_chosen     = jnp.where(any_accepted, f_trials[first_idx], f_x)
    return alpha_chosen, f_chosen


def _bfgs_scan(theta_raw0, f, grad_f, max_steps, n_halvings=20):
    """Fixed-iteration BFGS. Returns (theta_final, f_final).

    Implementation notes:
      - Initial H0 scaled by 1/||g0|| so first step has unit norm.
      - Nocedal & Wright "self-scaling" on first BFGS update: after we get
        the first valid (s, y), scale H by (sy/yy) before the update.
      - `n_halvings=20` gives line-search alphas down to ~1e-6, enough
        for TBATS's large-initial-gradient objectives.
      - If direction is uphill (H lost positive-definiteness), reset p = -g.
    """
    d = theta_raw0.shape[0]
    g0 = grad_f(theta_raw0)
    g0_norm = jnp.linalg.norm(g0) + 1e-30
    H0 = jnp.eye(d, dtype=theta_raw0.dtype) / g0_norm

    def step(state, _):
        x, H, f_x, g, first = state
        p = -H @ g
        uphill = jnp.dot(g, p) >= 0
        p = jnp.where(uphill, -g / (jnp.linalg.norm(g) + 1e-30), p)

        alpha, f_new = _backtracking_line_search(f, x, p, f_x, g,
                                                 n_halvings=n_halvings)
        s = alpha * p
        x_new = x + s
        g_new = grad_f(x_new)
        y = g_new - g
        sy = jnp.dot(s, y)
        yy = jnp.dot(y, y)

        # First-step self-scaling (Nocedal & Wright eq. 6.20). Only applied
        # the very first time sy > 0, to adapt H from its arbitrary initial
        # scale to the problem's actual curvature.
        scale = jnp.where((sy > 1e-10) & first, sy / jnp.maximum(yy, 1e-30), 1.0)
        H_scaled = scale * H

        # Sherman-Morrison BFGS inverse update.
        rho = 1.0 / jnp.maximum(sy, 1e-12)
        I = jnp.eye(d, dtype=x.dtype)
        A = I - rho * jnp.outer(s, y)
        B = I - rho * jnp.outer(y, s)
        H_updated = A @ H_scaled @ B + rho * jnp.outer(s, s)

        accept = sy > 1e-10
        H_new = jnp.where(accept, H_updated, H)
        # After the first genuine step, drop out of "first" mode.
        first_new = first & ~accept

        return (x_new, H_new, f_new, g_new, first_new), None

    x0 = theta_raw0
    f0 = f(x0)
    init = (x0, H0, f0, g0, jnp.bool_(True))  # last flag = "still in first-step warmup"
    (x_final, _, f_final, _, _), _ = lax.scan(step, init, None, length=max_steps)
    return x_final, f_final


# --- Public API ---------------------------------------------------------------


def _build_loss_grad(spec: TBATSSpec, y,
                     admissibility_weight: float = 1.0,
                     admissibility_margin: float = 1e-4,
                     gamma_ridge: float = 1e6):
    """Build jit'd f(theta_raw) and grad(theta_raw) bound to y + spec."""
    y_j = jnp.asarray(y)

    def loss(theta_raw):
        theta = raw_to_natural(theta_raw, spec)
        return penalized_objective(
            y_j, theta, spec,
            admissibility_weight=admissibility_weight,
            admissibility_margin=admissibility_margin,
            gamma_ridge=gamma_ridge,
        )

    return loss, jax.grad(loss)


def fit_scan(
    y,
    spec: TBATSSpec,
    theta0: Optional[np.ndarray] = None,
    max_steps: int = 500,
    n_halvings: int = 20,
    admissibility_weight: float = 1.0,
    admissibility_margin: float = 1e-4,
    gamma_ridge: float = 1e6,
) -> FitResultScan:
    """Fit via fixed-iteration BFGS-in-scan. Single series, TPU-friendly."""
    y_np = np.asarray(y, dtype=np.float64)
    if theta0 is None:
        theta0 = init_theta(spec, y_np)
    theta_raw0 = natural_to_raw(theta0, spec)

    loss, grad = _build_loss_grad(
        spec, y_np,
        admissibility_weight=admissibility_weight,
        admissibility_margin=admissibility_margin,
        gamma_ridge=gamma_ridge,
    )

    @jax.jit
    def run(t0):
        return _bfgs_scan(t0, loss, grad, max_steps=max_steps,
                          n_halvings=n_halvings)

    t0 = time.perf_counter()
    x, f = run(jnp.asarray(theta_raw0))
    jax.block_until_ready(x)
    compile_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    x, f = run(jnp.asarray(theta_raw0))
    jax.block_until_ready(x)
    wall_time = time.perf_counter() - t0

    theta_raw = np.asarray(x)
    theta = np.asarray(raw_to_natural(jnp.asarray(theta_raw), spec))
    pen_obj = float(f)
    nll_clean = float(neg_log_likelihood(jnp.asarray(y_np), jnp.asarray(theta), spec))

    return FitResultScan(
        theta=theta,
        theta_raw=theta_raw,
        neg_log_lik=pen_obj,
        neg_log_lik_clean=nll_clean,
        penalty=pen_obj - nll_clean,
        max_steps=max_steps,
        wall_time=wall_time,
        compile_time=compile_time,
        spec=spec,
    )


def fit_panel_scan(
    ys: np.ndarray,
    spec: TBATSSpec,
    max_steps: int = 500,
    n_halvings: int = 20,
    admissibility_weight: float = 1.0,
    admissibility_margin: float = 1e-4,
    gamma_ridge: float = 1e6,
) -> Tuple[np.ndarray, float, float]:
    """Batched fixed-iteration BFGS across a panel. Returns
    (theta_raw: (N, d), compile_time, wall_time).
    """
    ys = np.asarray(ys, dtype=np.float64)
    N, T = ys.shape

    theta_raw_stack = np.stack(
        [natural_to_raw(init_theta(spec, ys[i]), spec) for i in range(N)],
        axis=0,
    )
    raw_stack = jnp.asarray(theta_raw_stack)
    y_j = jnp.asarray(ys)

    panel_has_missing = bool(np.isnan(ys).any())

    def loss_one(theta_raw, y_i):
        theta = raw_to_natural(theta_raw, spec)
        return penalized_objective(
            y_i, theta, spec,
            admissibility_weight=admissibility_weight,
            admissibility_margin=admissibility_margin,
            gamma_ridge=gamma_ridge,
            has_missing=panel_has_missing,
        )

    def fit_one(theta_raw_init, y_i):
        # Close over y_i so loss is a pure scalar fn of theta_raw.
        def f(tr): return loss_one(tr, y_i)
        g = jax.grad(f)
        x, f_val = _bfgs_scan(theta_raw_init, f, g,
                              max_steps=max_steps, n_halvings=n_halvings)
        return x, f_val

    batched = jax.jit(jax.vmap(fit_one, in_axes=(0, 0)))

    t0 = time.perf_counter()
    thetas_raw, _ = batched(raw_stack, y_j)
    jax.block_until_ready(thetas_raw)
    compile_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    thetas_raw, _ = batched(raw_stack, y_j)
    jax.block_until_ready(thetas_raw)
    wall_time = time.perf_counter() - t0

    return np.asarray(thetas_raw), compile_time, wall_time
