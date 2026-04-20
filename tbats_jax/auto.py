"""Auto k-vector search — greedy per-period descent from max_k, AIC-guided.

Mirrors `forecast::tbats` R/tbats.R lines 211-272 (linear-search branch for
max_k <= 6). For each seasonal period, start at max_k and decrement while
AIC improves. Reuses `fit_jax` per candidate; structural search is an outer
Python loop.

AIC definition matches R's fitTBATS.R line 582:
    aic = neg_log_lik_clean + 2 * (n_smooth + state_dim)
        = neg_log_lik_clean + 2 * spec.n_params
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple
import time

import numpy as np

from tbats_jax.spec import TBATSSpec
from tbats_jax.fit_jax import fit_jax, FitResultJax


@dataclass
class AutoResult:
    fit: FitResultJax
    spec: TBATSSpec
    k_vector: Tuple[int, ...]
    aic: float
    n_candidates_tried: int
    wall_time: float


def _aic(fit_result: FitResultJax) -> float:
    """R-compatible AIC on the clean (unpenalized) likelihood."""
    return fit_result.neg_log_lik_clean + 2.0 * fit_result.spec.n_params


def _spec_with_k(base: TBATSSpec, periods, k_vector) -> TBATSSpec:
    return TBATSSpec(
        seasonal=tuple((float(p), int(k)) for p, k in zip(periods, k_vector)),
        use_trend=base.use_trend,
        use_damping=base.use_damping,
        use_box_cox=base.use_box_cox,
    )


def _default_max_k(periods: Tuple[float, ...]) -> Tuple[int, ...]:
    """Mirror R fitTBATS.R / tbats.R lines 215-232.

    For each period m_i, start with cap = floor((m_i - 1) / 2). For i > 0,
    if any harmonic k of period i produces a fundamental frequency m_i / k
    that divides any earlier period, reduce the cap to k - 1. This prevents
    aliasing between the seasonal components.
    """
    caps: List[int] = []
    for i, m in enumerate(periods):
        cap = max(1, math.floor((m - 1) / 2))
        if i > 0:
            current = 2
            while current <= cap:
                if m % current != 0:
                    current += 1
                    continue
                latter = m / current
                if any(periods[j] % latter == 0 for j in range(i)):
                    cap = current - 1
                    break
                current += 1
        caps.append(cap)
    return tuple(caps)


def auto_fit_jax(
    y,
    periods: Tuple[float, ...],
    use_trend: bool = True,
    use_damping: bool = True,
    use_box_cox: bool = False,
    max_k_per_period: Optional[Tuple[int, ...]] = None,
    max_steps: int = 1000,
    verbose: bool = False,
    **fit_kwargs,
) -> AutoResult:
    """Greedy search over harmonic counts k_i for each seasonal period.

    Parameters
    ----------
    y : 1D array
    periods : tuple of seasonal periods (e.g. (24.0, 168.0))
    use_trend, use_damping, use_box_cox : model structure flags (fixed)
    max_k_per_period : cap for each period's k. Default uses the
        floor((m_i - 1) / 2) cap (matching R's max.k) capped at 6 for
        practicality (R's 3-point search for larger k is not implemented).
    verbose : print progress per candidate

    Returns AutoResult with the best fit + chosen k_vector.
    """
    y_np = np.asarray(y, dtype=np.float64)

    # Default max_k mirrors R tbats.R lines 215-232 (cap + alias reduction).
    if max_k_per_period is None:
        max_k_per_period = _default_max_k(periods)

    if len(max_k_per_period) != len(periods):
        raise ValueError("max_k_per_period must match len(periods)")

    # Seed at max-k for every period.
    k_vector = list(max_k_per_period)

    def fit_k(k_vec):
        spec = _spec_with_k(
            TBATSSpec(seasonal=tuple(zip(periods, k_vec)),
                      use_trend=use_trend, use_damping=use_damping,
                      use_box_cox=use_box_cox),
            periods, k_vec,
        )
        return fit_jax(y_np, spec, max_steps=max_steps, **fit_kwargs)

    t0 = time.perf_counter()
    n_tried = 0

    best_fit = fit_k(k_vector)
    n_tried += 1
    best_aic = _aic(best_fit)
    if verbose:
        print(f"[auto] k={k_vector}  AIC={best_aic:.2f}  (seed)")

    # Greedy per-period descent.
    for i in range(len(periods)):
        while k_vector[i] > 1:
            trial = list(k_vector)
            trial[i] = trial[i] - 1
            new_fit = fit_k(trial)
            n_tried += 1
            new_aic = _aic(new_fit)
            if verbose:
                print(f"[auto] k={trial}  AIC={new_aic:.2f}")
            if new_aic >= best_aic:
                break
            best_aic = new_aic
            best_fit = new_fit
            k_vector = trial

    wall = time.perf_counter() - t0
    return AutoResult(
        fit=best_fit,
        spec=best_fit.spec,
        k_vector=tuple(k_vector),
        aic=best_aic,
        n_candidates_tried=n_tried,
        wall_time=wall,
    )
