"""Diagnostic: does init quality explain the fit_lm cold-start gap on Taylor?

Hypothesis A: LM cold-start fails because the LM/hinge interaction is wrong.
Hypothesis B: LM cold-start fails because the naive x0 init lands in a bad
              basin; better init (R-style OLS x0) would close the gap.

Test:    fit_lm cold from {naive init, OLS init, OLS-after-warmup init}
                  and compare Taylor held-out MAE.
Baseline: fit_jax cold (known good — MAE ~1042). Same naive init as fit_lm.

If Hypothesis B holds, OLS init alone should drop fit_lm's MAE substantially.
If A holds, OLS init won't help much and we should pursue log-hinge for LM.

Run: python -m benchmarks.diag_init_quality
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
jax.config.update("jax_enable_x64", True)

import time
from pathlib import Path
import numpy as np
import jax.numpy as jnp

from tbats_jax import TBATSSpec, fit_jax, fit_lm, forecast
from tbats_jax.params import init_theta
from tbats_jax.seed_state import with_ols_x0, warmup_then_ols


TAYLOR_CSV = Path("data/taylor.csv")
H = 336


def report(name: str, theta_final: np.ndarray, y_tr: np.ndarray, y_te: np.ndarray, t_wall: float):
    pred = np.asarray(forecast(y_tr, jnp.asarray(theta_final), spec, H))
    mae = float(np.mean(np.abs(y_te - pred)))
    print(f"  {name:48s}  MAE={mae:7.1f}   wall={t_wall:6.1f}s")
    return mae


if __name__ == "__main__":
    y = np.loadtxt(TAYLOR_CSV)
    y_tr, y_te = y[:-H], y[-H:]

    spec = TBATSSpec(
        seasonal=((48.0, 3), (336.0, 5)),
        use_trend=True,
        use_damping=True,
    )

    print(f"Taylor: T_train={len(y_tr)}, T_test={len(y_te)}, h={H}")
    print(f"Spec: state_dim={spec.state_dim}  n_params={spec.n_params}\n")

    # Baseline: fit_jax cold-start (BFGS + log-hinge)
    t0 = time.perf_counter()
    r_jax = fit_jax(y_tr, spec, max_steps=1000)
    t = time.perf_counter() - t0
    mae_jax = report("fit_jax cold (BFGS, naive init)", r_jax.theta, y_tr, y_te, t)

    # Baseline+: fit_jax with OLS init
    theta0_naive = init_theta(spec, y_tr)
    theta0_ols_for_jax = with_ols_x0(theta0_naive, y_tr, spec)
    t0 = time.perf_counter()
    r_jax_ols = fit_jax(y_tr, spec, theta0=theta0_ols_for_jax, max_steps=1000)
    t = time.perf_counter() - t0
    mae_jax_ols = report("fit_jax cold (BFGS, OLS init)", r_jax_ols.theta, y_tr, y_te, t)

    # A: fit_lm cold from naive init
    t0 = time.perf_counter()
    r_lm_naive = fit_lm(y_tr, spec, max_steps=200)
    t = time.perf_counter() - t0
    mae_naive = report("fit_lm cold (LM, naive init)", r_lm_naive.theta, y_tr, y_te, t)

    # B: fit_lm cold from OLS init (no warmup)
    theta0_ols = theta0_ols_for_jax
    t0 = time.perf_counter()
    r_lm_ols = fit_lm(y_tr, spec, theta0=theta0_ols, max_steps=200)
    t = time.perf_counter() - t0
    mae_ols = report("fit_lm cold (LM, OLS init, no warmup)", r_lm_ols.theta, y_tr, y_te, t)

    # C: fit_lm cold from OLS-after-warmup init
    t_init = time.perf_counter()
    theta0_warm_ols = warmup_then_ols(theta0_naive, y_tr, spec, warmup_iters=20)
    t_init = time.perf_counter() - t_init
    print(f"  [OLS+20-step warmup init computed in {t_init:.2f}s]")
    t0 = time.perf_counter()
    r_lm_warm_ols = fit_lm(y_tr, spec, theta0=theta0_warm_ols, max_steps=200)
    t = time.perf_counter() - t0
    mae_warm = report("fit_lm cold (LM, OLS+warmup init)", r_lm_warm_ols.theta, y_tr, y_te, t)

    print("\n=== Summary ===")
    print(f"  fit_jax (BFGS, naive init)            MAE={mae_jax:.1f}")
    print(f"  fit_jax (BFGS, OLS init)              MAE={mae_jax_ols:.1f}     "
          f"(delta vs naive: {mae_jax_ols - mae_jax:+.1f})")
    print(f"  fit_lm  (LM,   naive init)            MAE={mae_naive:.1f}")
    print(f"  fit_lm  (LM,   OLS init)              MAE={mae_ols:.1f}     "
          f"(delta vs naive: {mae_ols - mae_naive:+.1f})")
    print(f"  fit_lm  (LM,   OLS+warmup init)       MAE={mae_warm:.1f}     "
          f"(delta vs naive: {mae_warm - mae_naive:+.1f})")
    print()
    print("Interpretation:")
    if mae_warm < mae_naive * 0.95:
        print("  ✓ OLS+warmup init meaningfully improves fit_lm cold-start.")
        print("    Hypothesis B (init quality matters) is supported.")
    elif mae_ols < mae_naive * 0.95:
        print("  ✓ OLS init alone improves fit_lm cold-start.")
        print("    Hypothesis B (init quality matters) is supported.")
    else:
        print("  ✗ Better init does NOT meaningfully help fit_lm cold-start.")
        print("    Hypothesis A (LM/hinge interaction is the culprit) is supported.")
        print("    → Recommend pursuing log-hinge residual for LM (Option 1).")
