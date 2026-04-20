"""Panel benchmark: JAX vmap fit vs R sequential fit.

For N synthetic series of length T, compare:
  1. JAX fit_panel (vmap) — single fused call
  2. R forecast:::fitSpecificTBATS, invoked N times via Rscript (sequential)
  3. JAX fit_jax loop — sanity-check baseline (no vmap)

Measures total wall time and mean SSR across the panel.

Run: python -m benchmarks.bench_panel_full [N] [T]
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
jax.config.update("jax_enable_x64", True)

import sys
import time
import numpy as np

from tbats_jax import TBATSSpec, fit_jax, fit_panel
from tbats_jax.kernel import neg_log_likelihood
from tbats_jax.transforms import raw_to_natural
import jax.numpy as jnp
from benchmarks.data import synthesize_two_seasonal
from benchmarks.bench_r import r_available, run_r_bench


def build_panel(n_series: int, n_obs: int, periods) -> np.ndarray:
    return np.stack(
        [synthesize_two_seasonal(n=n_obs, periods=periods, seed=s) for s in range(n_series)],
        axis=0,
    )


def jax_panel(panel, spec):
    thetas_raw, nlls, compile_t, wall = fit_panel(panel, spec, max_steps=1000)
    ssrs = []
    for i in range(panel.shape[0]):
        th = np.asarray(raw_to_natural(jnp.asarray(thetas_raw[i]), spec))
        nll = float(neg_log_likelihood(jnp.asarray(panel[i]), jnp.asarray(th), spec))
        ssrs.append(np.exp(nll / panel.shape[1]) * panel.shape[1])
    return {
        "compile_time": compile_t,
        "wall_time": wall,
        "per_series_ms": wall / panel.shape[0] * 1000,
        "mean_ssr": float(np.mean(ssrs)),
    }


def jax_loop(panel, spec):
    t0 = time.perf_counter()
    ssrs = []
    for i in range(panel.shape[0]):
        r = fit_jax(panel[i], spec, max_steps=1000)
        ssrs.append(np.exp(r.neg_log_lik_clean / panel.shape[1]) * panel.shape[1])
    wall = time.perf_counter() - t0
    return {
        "wall_time": wall,
        "per_series_ms": wall / panel.shape[0] * 1000,
        "mean_ssr": float(np.mean(ssrs)),
    }


def r_sequential(panel, periods, k_vector):
    t0 = time.perf_counter()
    ssrs = []
    for i in range(panel.shape[0]):
        r = run_r_bench(panel[i], "fixed", True, True, periods, k_vector)
        ssrs.append(r["ssr"])
    wall = time.perf_counter() - t0
    return {
        "wall_time": wall,
        "per_series_ms": wall / panel.shape[0] * 1000,
        "mean_ssr": float(np.mean(ssrs)),
    }


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    T = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
    periods = (24.0, 168.0)
    k_vector = (3, 5)
    spec = TBATSSpec(
        seasonal=tuple(zip(periods, k_vector)),
        use_trend=True,
        use_damping=True,
    )

    print(f"\n=== Panel benchmark: N={N} series x T={T} obs ===")
    print(f"state_dim={spec.state_dim}  n_params={spec.n_params}")

    panel = build_panel(N, T, periods)

    print("\n-- JAX fit_panel (vmap) --")
    jv = jax_panel(panel, spec)
    print(f"  compile: {jv['compile_time']:.2f}s  warm wall: {jv['wall_time']:.3f}s")
    print(f"  per-series: {jv['per_series_ms']:.1f} ms   mean SSR: {jv['mean_ssr']:.2f}")

    print("\n-- JAX fit_jax (Python loop, no vmap) --")
    jl = jax_loop(panel, spec)
    print(f"  total: {jl['wall_time']:.2f}s   per-series: {jl['per_series_ms']:.1f} ms   mean SSR: {jl['mean_ssr']:.2f}")

    rs = None
    if r_available():
        print(f"\n-- R forecast sequential ({N} Rscript calls) --")
        # Only bench R on small N (Rscript startup is costly)
        if N <= 64:
            rs = r_sequential(panel, periods, k_vector)
            print(f"  total: {rs['wall_time']:.2f}s   per-series: {rs['per_series_ms']:.1f} ms   mean SSR: {rs['mean_ssr']:.2f}")
        else:
            print(f"  skipped (N={N} > 64; Rscript startup overhead would dominate)")
    else:
        print("\n-- R: SKIPPED (Rscript not on PATH) --")

    print("\n" + "=" * 60)
    print(f"{'Backend':<28} {'Total (s)':>10} {'Per-series (ms)':>17} {'Mean SSR':>10}")
    print("-" * 60)
    print(f"{'JAX fit_panel (vmap)':<28} {jv['wall_time']:>10.3f} {jv['per_series_ms']:>17.2f} {jv['mean_ssr']:>10.2f}")
    print(f"{'JAX fit_jax loop':<28} {jl['wall_time']:>10.3f} {jl['per_series_ms']:>17.2f} {jl['mean_ssr']:>10.2f}")
    if rs is not None:
        print(f"{'R forecast sequential':<28} {rs['wall_time']:>10.3f} {rs['per_series_ms']:>17.2f} {rs['mean_ssr']:>10.2f}")

    print()
    if rs is not None:
        print(f"vmap speedup vs R sequential : {rs['wall_time'] / jv['wall_time']:.1f}x")
    print(f"vmap speedup vs JAX loop     : {jl['wall_time'] / jv['wall_time']:.1f}x")


if __name__ == "__main__":
    main()
