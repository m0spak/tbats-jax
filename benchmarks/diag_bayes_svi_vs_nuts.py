"""Diagnostic: SVI vs NUTS on a moderately hard Bayesian TBATS problem.

Problem: T=500, dual seasonal (24, 168) with K=(2, 2), trend + damping.
State_dim ~12, parameter dim ~20. Big enough that NUTS' step-size
collapse near the log-hinge admissibility barrier becomes likely.

Reports for each method:
  - wall time
  - posterior mean of alpha, beta, phi, sigma (for sanity)
  - posterior std of alpha (a stand-in for "did it actually mix?")
  - For NUTS: number of effective samples (or step-size proxy)
  - For SVI: final ELBO, ELBO trajectory shape

Run: python -m benchmarks.diag_bayes_svi_vs_nuts
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
jax.config.update("jax_enable_x64", True)

import time
import numpy as np

from tbats_jax import TBATSSpec, bayes_tbats, svi_tbats


def synth(T: int = 500, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    y = (
        20.0 + 0.005 * t
        + 1.5 * np.sin(2 * np.pi * t / 24.0)
        + 0.8 * np.cos(2 * np.pi * t / 24.0)
        + 0.6 * np.sin(2 * np.pi * t / 168.0)
        + rng.normal(0, 0.4, T)
    )
    return y


def report(name: str, samples: dict, wall: float):
    a = samples["alpha"]
    b = samples.get("beta", np.zeros_like(a))
    s = samples["sigma"]
    print(f"  {name}")
    print(f"    wall:           {wall:7.2f} s")
    print(f"    alpha:   mean={float(a.mean()):+8.4f}  std={float(a.std()):.4f}")
    print(f"    beta:    mean={float(b.mean()):+8.4f}  std={float(b.std()):.4f}")
    print(f"    sigma:   mean={float(s.mean()):+8.4f}  std={float(s.std()):.4f}")
    if a.std() < 1e-8:
        print("    !! posterior std on alpha collapsed → sampler stuck at single point")


if __name__ == "__main__":
    y = synth(T=500)
    spec = TBATSSpec(
        seasonal=((24.0, 2), (168.0, 2)),
        use_trend=True,
        use_damping=True,
    )
    print(f"Problem: T={len(y)}, state_dim={spec.state_dim}, "
          f"n_params={spec.n_params}\n")

    # NUTS run — small budget so we don't wait too long if it gets stuck
    print("=== NUTS (bayes_tbats) ===")
    t0 = time.perf_counter()
    nuts_res = bayes_tbats(y, spec, num_warmup=200, num_samples=200, num_chains=1)
    nuts_wall = time.perf_counter() - t0
    report("NUTS  (200 warmup, 200 samples)", nuts_res.samples, nuts_wall)

    # SVI run
    print("\n=== SVI (svi_tbats, AutoNormal mean-field) ===")
    t0 = time.perf_counter()
    svi_res = svi_tbats(y, spec, num_steps=2000, num_posterior_samples=200)
    svi_wall = time.perf_counter() - t0
    report("SVI   (2000 steps, 200 posterior samples)", svi_res.samples, svi_wall)

    print("\n=== Verdict ===")
    a_nuts_std = float(nuts_res.samples["alpha"].std())
    a_svi_std = float(svi_res.samples["alpha"].std())
    print(f"  alpha posterior std — NUTS: {a_nuts_std:.6f}, SVI: {a_svi_std:.6f}")
    if a_nuts_std < 1e-4 and a_svi_std > 1e-3:
        print("  ✓ NUTS posterior collapsed (stuck), SVI gave a real posterior.")
    elif a_nuts_std > 1e-3 and a_svi_std > 1e-3:
        print("  ✓ Both produced non-trivial posteriors; SVI is the faster path.")
    elif a_svi_std < 1e-4:
        print("  ✗ SVI posterior also collapsed — variational family too narrow,")
        print("    or the optimizer itself got pushed to the boundary.")
    else:
        print("  ? Mixed result — inspect manually.")
