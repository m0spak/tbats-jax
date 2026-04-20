"""Real-dataset benchmark: forecast::taylor (half-hourly electricity).

n=4032, two seasonal periods (48, 336) — the canonical TBATS workload from
De Livera et al. (2011). Run ./benchmarks/fetch_real_data.R first to write
data/taylor.csv.

Run: python -m benchmarks.bench_real
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
jax.config.update("jax_enable_x64", True)

import json
import time
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from tbats_jax import TBATSSpec, fit_jax, forecast
from benchmarks.bench_r import r_available, run_r_bench

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"


def load_taylor():
    csv = DATA_DIR / "taylor.csv"
    meta = DATA_DIR / "taylor.json"
    if not csv.exists():
        raise FileNotFoundError(
            f"{csv} not found. Run: Rscript benchmarks/fetch_real_data.R data"
        )
    y = np.loadtxt(csv)
    m = json.loads(meta.read_text())
    return y, tuple(float(p) for p in m["periods"])


def accuracy(pred, actual):
    err = actual - pred
    return {
        "mae":  float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
    }


def main():
    y, periods = load_taylor()
    # De Livera et al. used k=(3, 5) for taylor; match that for fair comparison.
    k_vector = (3, 5)
    h = 336  # one week held out
    y_tr, y_te = y[:-h], y[-h:]

    spec = TBATSSpec(
        seasonal=tuple(zip(periods, k_vector)),
        use_trend=True,
        use_damping=True,
    )

    print(f"\n=== bench_real: forecast::taylor  n={len(y)}  periods={periods} ===")
    print(f"  k_vector={k_vector}  state_dim={spec.state_dim}  n_params={spec.n_params}")
    print(f"  train={len(y_tr)}  test={h} (one week held out)")
    print(f"  y range: [{y.min():.0f}, {y.max():.0f}]  mean={y.mean():.0f}")

    print("\n-- JAX (optimistix) --")
    t0 = time.perf_counter()
    r = fit_jax(y_tr, spec, max_steps=1000)
    pred = np.asarray(forecast(y_tr, jnp.asarray(r.theta), spec, h))
    jax_total = time.perf_counter() - t0
    ja = accuracy(pred, y_te)
    ssr = float(np.exp(r.neg_log_lik_clean / len(y_tr)) * len(y_tr))
    print(f"  fit_wall={r.wall_time:.3f}s  compile={r.compile_time:.2f}s  total={jax_total:.2f}s")
    print(f"  train SSR={ssr:.2f}  penalty={r.penalty:.2e}")
    print(f"  test MAE={ja['mae']:.2f}  RMSE={ja['rmse']:.2f}")

    if r_available():
        print("\n-- R forecast::tbats (fixed k) --")
        try:
            r_fixed = run_r_bench(y_tr, "fixed", True, True, periods, k_vector, h=h,
                                  timeout_s=900)
            r_pred = np.asarray(r_fixed["forecast"])
            ra = accuracy(r_pred, y_te)
            print(f"  wall={r_fixed['wall_time_s']:.2f}s  SSR={r_fixed['ssr']:.2f}")
            print(f"  test MAE={ra['mae']:.2f}  RMSE={ra['rmse']:.2f}")
        except Exception as exc:
            print(f"  R fixed FAIL: {exc}")
            ra = None

        print("\n-- R forecast::tbats (auto) --")
        try:
            r_auto = run_r_bench(y_tr, "auto", True, True, periods, k_vector, h=h,
                                 timeout_s=1800)
            r_auto_pred = np.asarray(r_auto["forecast"])
            ra_acc = accuracy(r_auto_pred, y_te)
            print(f"  wall={r_auto['wall_time_s']:.2f}s  SSR={r_auto['ssr']:.2f}  chose k={r_auto['chosen_k']}")
            print(f"  test MAE={ra_acc['mae']:.2f}  RMSE={ra_acc['rmse']:.2f}")
        except Exception as exc:
            print(f"  R auto FAIL: {exc}")
            ra_acc = None
    else:
        ra = ra_acc = None
        print("\n-- R: SKIPPED (Rscript not on PATH) --")

    print("\n" + "=" * 70)
    print(f"{'Backend':<28} {'Wall (s)':>10} {'Train SSR':>12} {'Test MAE':>10} {'Test RMSE':>10}")
    print("-" * 70)
    print(f"{'JAX (optimistix)':<28} {jax_total:>10.2f} {ssr:>12.2f} {ja['mae']:>10.2f} {ja['rmse']:>10.2f}")
    if ra is not None:
        print(f"{'R forecast fixed':<28} {r_fixed['wall_time_s']:>10.2f} {r_fixed['ssr']:>12.2f} {ra['mae']:>10.2f} {ra['rmse']:>10.2f}")
    if ra_acc is not None:
        print(f"{'R forecast auto (k=' + str(r_auto['chosen_k']) + ')':<28} {r_auto['wall_time_s']:>10.2f} {r_auto['ssr']:>12.2f} {ra_acc['mae']:>10.2f} {ra_acc['rmse']:>10.2f}")


if __name__ == "__main__":
    main()
