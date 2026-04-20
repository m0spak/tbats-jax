"""Out-of-sample forecast accuracy benchmark.

Splits the series into train (first 90%) and test (last 10%), fits each
backend on train, produces h-step point forecasts, then reports MAE/RMSE
against the held-out tail. Complements bench_single.py which measures
in-sample fit quality (SSR) instead.

Run: python -m benchmarks.bench_oos
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

from tbats_jax import TBATSSpec, fit_jax, forecast
from benchmarks.data import synthesize_two_seasonal
from benchmarks.bench_r import r_available, run_r_bench


def accuracy(pred: np.ndarray, actual: np.ndarray) -> dict:
    err = actual - pred
    return {
        "mae":  float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
    }


def jax_forecast(y_train, spec: TBATSSpec, h: int) -> tuple:
    """Fit on train, return (h-step forecast, wall_time, ssr)."""
    res = fit_jax(y_train, spec, max_steps=1000)
    preds = np.asarray(forecast(y_train, jnp.asarray(res.theta), spec, h))
    ssr = float(np.exp(res.neg_log_lik_clean / len(y_train)) * len(y_train))
    return preds, res.wall_time, ssr


def tbats_forecast(y_train, periods, use_trend, use_damping, h: int):
    """Fit Python tbats (full auto-search), return (forecast, wall_time)."""
    import time as _time
    from tbats import TBATS
    est = TBATS(
        seasonal_periods=list(periods),
        use_trend=use_trend,
        use_damped_trend=use_damping,
        use_arma_errors=False,
        use_box_cox=False,
        n_jobs=1,
    )
    t0 = _time.perf_counter()
    model = est.fit(y_train)
    wall = _time.perf_counter() - t0
    preds = np.asarray(model.forecast(steps=h))
    return preds, wall


def main():
    periods = (24.0, 168.0)
    k_vector = (3, 5)
    spec = TBATSSpec(
        seasonal=tuple(zip(periods, k_vector)),
        use_trend=True,
        use_damping=True,
    )
    n = 1500
    h = 150  # last 10%
    y = synthesize_two_seasonal(n=n, periods=periods)
    y_train, y_test = y[:-h], y[-h:]

    print(f"\n=== OOS bench: n_train={len(y_train)}  h={h}  periods={periods}  k={k_vector} ===")

    results = []

    print("\n-- JAX --")
    jax_pred, jax_wall, jax_ssr = jax_forecast(y_train, spec, h)
    jax_acc = accuracy(jax_pred, y_test)
    results.append(("JAX (fixed k)", jax_wall, jax_ssr, jax_acc))
    print(f"  wall={jax_wall:.3f}s  SSR={jax_ssr:.2f}  MAE={jax_acc['mae']:.4f}  RMSE={jax_acc['rmse']:.4f}")

    print("\n-- Python `tbats` (auto) --")
    try:
        tb_pred, tb_wall = tbats_forecast(y_train, periods, True, True, h)
        # ssr on train from fit residuals — get via full bench or skip; approximate here by leaving None
        tb_acc = accuracy(tb_pred, y_test)
        results.append(("Py tbats auto", tb_wall, None, tb_acc))
        print(f"  wall={tb_wall:.3f}s  MAE={tb_acc['mae']:.4f}  RMSE={tb_acc['rmse']:.4f}")
    except Exception as exc:
        print(f"  FAIL: {exc}")

    if r_available():
        print("\n-- R forecast::tbats (fixed k) --")
        try:
            r = run_r_bench(y_train, "fixed", True, True, periods, k_vector, h=h)
            r_pred = np.asarray(r["forecast"])
            r_acc = accuracy(r_pred, y_test)
            results.append(("R forecast fixed", r["wall_time_s"], r["ssr"], r_acc))
            print(f"  wall={r['wall_time_s']:.3f}s  SSR={r['ssr']:.2f}  MAE={r_acc['mae']:.4f}  RMSE={r_acc['rmse']:.4f}")
        except Exception as exc:
            print(f"  R fixed FAIL: {exc}")

        print("\n-- R forecast::tbats (auto) --")
        try:
            ra = run_r_bench(y_train, "auto", True, True, periods, k_vector, h=h)
            ra_pred = np.asarray(ra["forecast"])
            ra_acc = accuracy(ra_pred, y_test)
            results.append((f"R forecast auto (k={ra['chosen_k']})", ra["wall_time_s"], ra["ssr"], ra_acc))
            print(f"  wall={ra['wall_time_s']:.3f}s  SSR={ra['ssr']:.2f}  MAE={ra_acc['mae']:.4f}  RMSE={ra_acc['rmse']:.4f}")
        except Exception as exc:
            print(f"  R auto FAIL: {exc}")
    else:
        print("\n-- R: SKIPPED (Rscript not on PATH) --")

    print("\n" + "=" * 72)
    print(f"{'Backend':<30} {'Wall (s)':>10} {'Train SSR':>12} {'Test MAE':>10} {'Test RMSE':>10}")
    print("-" * 72)
    for name, wall, ssr, acc in results:
        ssr_s = f"{ssr:.2f}" if ssr is not None else "—"
        print(f"{name:<30} {wall:>10.3f} {ssr_s:>12} {acc['mae']:>10.4f} {acc['rmse']:>10.4f}")


if __name__ == "__main__":
    main()
