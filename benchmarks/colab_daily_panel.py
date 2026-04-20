"""Daily-panel GPU benchmark — the realistic TBATS workload.

T=730 (2 years daily) with weekly + yearly seasonality. Per-series CPU
time is ~45 ms, so the GPU has a real shot at winning at moderate N.
This is the workload shape most TBATS deployments actually see.

Usage (local, CPU): `.venv/bin/python -m benchmarks.colab_daily_panel [N]`
Usage (Colab GPU): call `run(N=...)` from the notebook.
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "")
import jax
jax.config.update("jax_enable_x64", True)

import sys
import time
import numpy as np
import jax.numpy as jnp

from tbats_jax import TBATSSpec, fit_panel
from tbats_jax.transforms import raw_to_natural
from tbats_jax.kernel import neg_log_likelihood
from benchmarks.data import synthesize_daily


def build_panel(N, T):
    return np.stack([synthesize_daily(n=T, seed=s) for s in range(N)], axis=0)


def run(N: int = 1000, T: int = 730, max_steps: int = 500):
    spec = TBATSSpec(
        seasonal=((7.0, 3), (365.25, 5)),
        use_trend=True,
        use_damping=True,
    )
    dev = jax.devices()[0]
    backend = jax.default_backend()
    print("=" * 62)
    print(f"JAX backend: {backend}  ({dev})")
    print(f"Daily panel: N={N}  T={T}  state_dim={spec.state_dim}  "
          f"n_params={spec.n_params}")
    print("=" * 62)

    panel = build_panel(N, T)
    t0 = time.perf_counter()
    thetas_raw, nlls, compile_t, warm_wall = fit_panel(
        panel, spec, max_steps=max_steps,
    )
    jax.block_until_ready(thetas_raw)
    total = time.perf_counter() - t0

    # Sanity: recover SSR for a few series
    ssrs = []
    for i in range(min(N, 20)):
        th = np.asarray(raw_to_natural(jnp.asarray(thetas_raw[i]), spec))
        nll = float(neg_log_likelihood(jnp.asarray(panel[i]), jnp.asarray(th), spec))
        ssrs.append(np.exp(nll / T) * T)
    mean_ssr = float(np.mean(ssrs))

    print(f"\ncompile time     : {compile_t:.2f} s (one-time)")
    print(f"warm panel fit   : {warm_wall:.2f} s")
    print(f"per-series (warm): {warm_wall / N * 1000:.2f} ms")
    print(f"mean SSR (20)    : {mean_ssr:.2f}")
    return {
        "backend": backend,
        "N": N, "T": T,
        "compile_time": compile_t,
        "warm_wall": warm_wall,
        "per_series_ms": warm_wall / N * 1000,
        "mean_ssr_20": mean_ssr,
    }


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    T = int(sys.argv[2]) if len(sys.argv) > 2 else 730
    run(N, T)
