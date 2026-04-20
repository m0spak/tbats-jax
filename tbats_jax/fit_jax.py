"""JAX-native fit via optimistix. End-to-end on-graph: jit + vmap-friendly.

Entry point: fit_jax(y, spec, ...) -> FitResultJax
Batched:     fit_panel(ys, spec, ...) -> arrays of thetas / NLLs over a panel
"""

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx

from tbats_jax.spec import TBATSSpec
from tbats_jax.params import init_theta, unpack
from tbats_jax.kernel import penalized_objective, neg_log_likelihood
from tbats_jax.transforms import raw_to_natural, natural_to_raw


@dataclass
class FitResultJax:
    theta: np.ndarray            # natural (unpacked) parameters
    theta_raw: np.ndarray        # unconstrained optimizer variable
    neg_log_lik: float           # penalized objective at solution
    neg_log_lik_clean: float     # pure NLL (no penalty)
    penalty: float
    n_steps: int
    wall_time: float
    compile_time: float
    spec: TBATSSpec


def _build_loss(spec: TBATSSpec, y,
                admissibility_weight: float = 1.0,
                admissibility_margin: float = 1e-4,
                gamma_ridge: float = 1e6,
                has_missing: bool = False):
    """Return a pure scalar loss function loss(theta_raw, args) suitable for
    optimistix.minimise. Works on untransformed `theta_raw` in R^n.
    """
    y_j = jnp.asarray(y)

    def loss(theta_raw, args):
        theta = raw_to_natural(theta_raw, spec)
        return penalized_objective(
            y_j, theta, spec,
            admissibility_weight=admissibility_weight,
            admissibility_margin=admissibility_margin,
            gamma_ridge=gamma_ridge,
            has_missing=has_missing,
        )

    return loss


def fit_jax(
    y,
    spec: TBATSSpec,
    theta0: Optional[np.ndarray] = None,
    max_steps: int = 1000,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    admissibility_weight: float = 1.0,
    admissibility_margin: float = 1e-4,
    gamma_ridge: float = 1e6,
) -> FitResultJax:
    """Fit using optimistix BFGS. Single series.

    theta0 is in the NATURAL parameter space (same as scipy fit). Converted
    internally to raw unconstrained space for the optimizer.
    """
    y_np = np.asarray(y, dtype=np.float64)
    if theta0 is None:
        theta0 = init_theta(spec, y_np)
    theta_raw0 = natural_to_raw(theta0, spec)

    has_missing = bool(np.isnan(y_np).any())
    loss = _build_loss(spec, y_np,
                       admissibility_weight=admissibility_weight,
                       admissibility_margin=admissibility_margin,
                       gamma_ridge=gamma_ridge,
                       has_missing=has_missing)
    solver = optx.BFGS(rtol=rtol, atol=atol)

    @jax.jit
    def run(theta_raw_init):
        return optx.minimise(
            loss, solver, theta_raw_init,
            max_steps=max_steps, throw=False,
        )

    t0 = time.perf_counter()
    sol = run(jnp.asarray(theta_raw0))
    jax.block_until_ready(sol.value)
    compile_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    sol = run(jnp.asarray(theta_raw0))
    jax.block_until_ready(sol.value)
    wall_time = time.perf_counter() - t0

    theta_raw = np.asarray(sol.value)
    theta = np.asarray(raw_to_natural(jnp.asarray(theta_raw), spec))
    pen_obj = float(loss(jnp.asarray(theta_raw), None))
    nll_clean = float(neg_log_likelihood(jnp.asarray(y_np), jnp.asarray(theta), spec,
                                          has_missing=has_missing))

    return FitResultJax(
        theta=theta,
        theta_raw=theta_raw,
        neg_log_lik=pen_obj,
        neg_log_lik_clean=nll_clean,
        penalty=pen_obj - nll_clean,
        n_steps=int(getattr(sol.stats, "num_steps", 0)) if hasattr(sol, "stats") else -1,
        wall_time=wall_time,
        compile_time=compile_time,
        spec=spec,
    )


def fit_panel_hetero(
    series_and_specs: list,
    max_steps: int = 1000,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    admissibility_weight: float = 1.0,
    admissibility_margin: float = 1e-4,
    gamma_ridge: float = 1e6,
) -> list:
    """Fit a heterogeneous panel: list of (y, TBATSSpec) with possibly
    different structures and lengths.

    Strategy: bucket series by spec identity; within each bucket pad all
    series to the max length with NaN and call fit_panel on the stack. The
    NaN-masked scan handles different effective series lengths transparently.

    Returns a list of dicts, one per input (in original order), each with
    {"theta": np.ndarray, "neg_log_lik_clean": float, "wall_time": float,
     "compile_time": float, "bucket_size": int}.
    """
    from collections import defaultdict

    n = len(series_and_specs)
    buckets: dict = defaultdict(list)
    for idx, (y, spec) in enumerate(series_and_specs):
        buckets[spec].append((idx, np.asarray(y, dtype=np.float64)))

    out_by_idx: dict = {}
    for spec, items in buckets.items():
        T_max = max(len(y) for _, y in items)
        ys_padded = np.full((len(items), T_max), np.nan, dtype=np.float64)
        for i, (_, y) in enumerate(items):
            ys_padded[i, : len(y)] = y

        thetas_raw, nlls, compile_t, wall = fit_panel(
            ys_padded, spec,
            max_steps=max_steps, rtol=rtol, atol=atol,
            admissibility_weight=admissibility_weight,
            admissibility_margin=admissibility_margin,
            gamma_ridge=gamma_ridge,
        )
        for i, (orig_idx, y) in enumerate(items):
            theta_nat = np.asarray(raw_to_natural(jnp.asarray(thetas_raw[i]), spec))
            has_missing = bool(np.isnan(y).any())
            nll_clean = float(neg_log_likelihood(jnp.asarray(y), jnp.asarray(theta_nat), spec,
                                                 has_missing=has_missing))
            out_by_idx[orig_idx] = {
                "theta": theta_nat,
                "neg_log_lik_clean": nll_clean,
                "wall_time": wall,
                "compile_time": compile_t,
                "bucket_size": len(items),
                "spec": spec,
            }

    return [out_by_idx[i] for i in range(n)]


def fit_panel(
    ys: np.ndarray,
    spec: TBATSSpec,
    theta0: Optional[np.ndarray] = None,
    max_steps: int = 1000,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    admissibility_weight: float = 1.0,
    admissibility_margin: float = 1e-4,
    gamma_ridge: float = 1e6,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Batched fit across a panel of series (N, T).

    All series must share the same TBATS structure. Returns:
      (thetas_raw: (N, d), nlls: (N,), compile_time, wall_time)
    """
    ys = np.asarray(ys, dtype=np.float64)
    N, T = ys.shape

    if theta0 is None:
        # Per-series init (level = mean of first 10 obs differs across series).
        theta_raw_stack = np.stack(
            [natural_to_raw(init_theta(spec, ys[i]), spec) for i in range(N)],
            axis=0,
        )
    else:
        theta_raw0 = natural_to_raw(theta0, spec)
        theta_raw_stack = np.broadcast_to(theta_raw0, (N, len(theta_raw0))).copy()
    raw_stack = jnp.asarray(theta_raw_stack)

    solver = optx.BFGS(rtol=rtol, atol=atol)

    # For a batched panel, treat as potentially missing if ANY series has
    # NaN — same scan graph is used for all rows. Avoids mixing graphs per
    # row while still handling padding from fit_panel_hetero.
    panel_has_missing = bool(np.isnan(ys).any())

    def loss_one(theta_raw, args):
        y_i = args
        theta = raw_to_natural(theta_raw, spec)
        return penalized_objective(
            y_i, theta, spec,
            admissibility_weight=admissibility_weight,
            admissibility_margin=admissibility_margin,
            gamma_ridge=gamma_ridge,
            has_missing=panel_has_missing,
        )

    def fit_one(theta_raw_init, y_i):
        sol = optx.minimise(
            loss_one, solver, theta_raw_init,
            args=y_i, max_steps=max_steps, throw=False,
        )
        return sol.value, loss_one(sol.value, y_i)

    batched = jax.jit(jax.vmap(fit_one, in_axes=(0, 0)))

    y_j = jnp.asarray(ys)
    t0 = time.perf_counter()
    thetas_raw, nlls = batched(raw_stack, y_j)
    jax.block_until_ready(thetas_raw)
    compile_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    thetas_raw, nlls = batched(raw_stack, y_j)
    jax.block_until_ready(thetas_raw)
    wall_time = time.perf_counter() - t0

    return np.asarray(thetas_raw), np.asarray(nlls), compile_time, wall_time
