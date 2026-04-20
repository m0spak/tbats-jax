"""Colab-ready panel benchmark. Runs identically on CPU and GPU.

Usage (local): `.venv/bin/python -m benchmarks.colab_panel_gpu [N] [T]`
Usage (Colab): upload this directory (or `!pip install -e <your repo>`),
switch runtime to GPU, then:

    !pip install -q jax[cuda12] optimistix scipy
    !pip install -q -e /content/tbats_jax
    import os; os.environ['JAX_PLATFORMS'] = ''  # allow CUDA
    import jax; jax.config.update("jax_enable_x64", True)
    !python -m benchmarks.colab_panel_gpu 500 1500

The same fit_panel call fuses N series into one JIT'd scan + BFGS on CPU;
on GPU the scan lanes plus batched linear algebra typically give a further
10-100x speedup for large N.
"""

import os
# Do NOT force CPU here — let the runtime pick. On a Colab GPU runtime,
# jax will auto-detect CUDA; locally it falls back to CPU.
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
from benchmarks.data import synthesize_two_seasonal


def device_info():
    devs = jax.devices()
    return {
        "backend": jax.default_backend(),
        "count": len(devs),
        "name": str(devs[0]),
    }


def build_panel(N, T, periods):
    return np.stack(
        [synthesize_two_seasonal(n=T, periods=periods, seed=s) for s in range(N)],
        axis=0,
    )


def run(N: int, T: int):
    periods = (24.0, 168.0)
    k_vector = (3, 5)
    spec = TBATSSpec(
        seasonal=tuple(zip(periods, k_vector)),
        use_trend=True,
        use_damping=True,
    )

    info = device_info()
    print("=" * 60)
    print(f"JAX backend: {info['backend']}  ({info['count']}x {info['name']})")
    print(f"Panel:  N={N} series  T={T} obs  state_dim={spec.state_dim}  n_params={spec.n_params}")
    print("=" * 60)

    panel = build_panel(N, T, periods)

    print("\nbuilding + compiling... ", end="", flush=True)
    t0 = time.perf_counter()
    thetas_raw, nlls, compile_t, warm_wall = fit_panel(panel, spec, max_steps=1000)
    jax.block_until_ready(thetas_raw)
    total = time.perf_counter() - t0
    print("done.")

    # Accuracy sanity
    ssrs = []
    for i in range(min(N, 20)):
        th = np.asarray(raw_to_natural(jnp.asarray(thetas_raw[i]), spec))
        nll = float(neg_log_likelihood(jnp.asarray(panel[i]), jnp.asarray(th), spec))
        ssrs.append(np.exp(nll / T) * T)
    mean_ssr = float(np.mean(ssrs))

    print(f"\ncompile time     : {compile_t:.2f} s (one-time JIT)")
    print(f"warm panel fit   : {warm_wall:.2f} s")
    print(f"per-series (warm): {warm_wall / N * 1000:.2f} ms")
    print(f"mean SSR (20)    : {mean_ssr:.2f}")
    return {
        "backend": info["backend"],
        "compile_time": compile_t,
        "warm_wall": warm_wall,
        "per_series_ms": warm_wall / N * 1000,
        "mean_ssr_20": mean_ssr,
    }


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    T = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
    run(N, T)
