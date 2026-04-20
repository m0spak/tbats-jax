"""Single-series benchmark: JAX TBATS vs Python `tbats` package.

Both fit the same fixed structure (trend + two seasonals, no Box-Cox, no ARMA)
so we can compare fit time and in-sample SSR on equal terms.

Run: python -m benchmarks.bench_single
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
jax.config.update("jax_enable_x64", True)

import time
import numpy as np

from tbats_jax import TBATSSpec, fit, fit_jax, forecast
from tbats_jax.kernel import sse
from benchmarks.data import synthesize_two_seasonal
from benchmarks.bench_r import r_available, run_r_bench


def bench_jax_scipy(y, spec):
    r = fit(y, spec, maxiter=500)
    ssr = float(np.exp(r.neg_log_lik_clean / len(y)) * len(y))
    return {"ssr": ssr, "wall_time": r.wall_time, "penalty": r.penalty}


def bench_jax_optx(y, spec):
    r = fit_jax(y, spec, max_steps=1000)
    ssr = float(np.exp(r.neg_log_lik_clean / len(y)) * len(y))
    return {
        "ssr": ssr,
        "wall_time": r.wall_time,
        "compile_time": r.compile_time,
        "penalty": r.penalty,
    }


def bench_tbats_auto(y, periods, use_trend, use_damping):
    """Python `tbats` package with full auto-search (fair to its design)."""
    from tbats import TBATS

    est = TBATS(
        seasonal_periods=list(periods),
        use_trend=use_trend,
        use_damped_trend=use_damping,
        use_arma_errors=False,
        use_box_cox=False,
        n_jobs=1,
    )
    t0 = time.perf_counter()
    model = est.fit(y)
    wall_time = time.perf_counter() - t0

    resid = np.asarray(model.resid)
    ssr = float(np.sum(resid ** 2))
    n = len(y)
    neg_ll = n * float(np.log(ssr / n)) if ssr > 0 else float("-inf")
    return {
        "neg_log_lik": neg_ll,
        "ssr": ssr,
        "wall_time": wall_time,
        "aic": float(model.aic),
        "chosen_harmonics": getattr(model.params.components, "seasonal_harmonics", None),
    }


def bench_tbats_fixed(y, periods, use_trend, use_damping, k_vector):
    """Pin tbats to the same k-vector for apples-to-apples structure parity.

    Uses internal API (_choose_model_from_possible_component_settings).
    """
    from tbats import TBATS
    from sklearn.model_selection import ParameterGrid

    est = TBATS(
        seasonal_periods=list(periods),
        use_trend=use_trend,
        use_damped_trend=use_damping,
        use_arma_errors=False,
        use_box_cox=False,
        n_jobs=1,
    )
    # Reuse framework setup via a private call path
    est.seasonal_periods = est._normalize_seasonal_periods(list(periods))
    grid = ParameterGrid({
        "use_box_cox": [False],
        "use_trend": [use_trend],
        "use_damped_trend": [use_damping],
        "seasonal_harmonics": [list(k_vector)],
        "seasonal_periods": [list(est.seasonal_periods)],
        "use_arma_errors": [False],
    })
    t0 = time.perf_counter()
    model = est._choose_model_from_possible_component_settings(np.asarray(y), components_grid=grid)
    wall_time = time.perf_counter() - t0

    resid = np.asarray(model.resid)
    ssr = float(np.sum(resid ** 2))
    n = len(y)
    neg_ll = n * float(np.log(ssr / n))
    return {
        "neg_log_lik": neg_ll,
        "ssr": ssr,
        "wall_time": wall_time,
        "aic": float(model.aic),
    }


def main():
    periods = (24.0, 168.0)
    k_vector = (3, 5)
    spec = TBATSSpec(
        seasonal=tuple(zip(periods, k_vector)),
        use_trend=True,
        use_damping=True,
    )
    n = 1500
    y = synthesize_two_seasonal(n=n, periods=periods)

    print(f"\n=== Benchmark: n={n}, periods={periods}, k={k_vector} ===")
    print(f"state_dim={spec.state_dim}  n_params={spec.n_params}")

    print("\n-- JAX TBATS (optimistix BFGS) --")
    jo = bench_jax_optx(y, spec)
    for k, v in jo.items():
        print(f"  {k}: {v}")

    print("\n-- JAX TBATS (scipy L-BFGS-B) --")
    js = bench_jax_scipy(y, spec)
    for k, v in js.items():
        print(f"  {k}: {v}")
    # Keep legacy name for compat with later print code
    j = {"warm_wall_time": jo["wall_time"], "ssr": jo["ssr"],
         "first_compile_time": jo["compile_time"],
         "neg_log_lik_clean": float('nan'), "penalty": jo["penalty"]}

    print("\n-- Python `tbats` package (fixed structure, k pinned) --")
    try:
        tf = bench_tbats_fixed(y, periods, True, True, k_vector)
        for k, v in tf.items():
            print(f"  {k}: {v}")
    except Exception as exc:
        tf = None
        print(f"  tbats fixed failed: {exc}")

    print("\n-- Python `tbats` package (full auto-search) --")
    try:
        ta = bench_tbats_auto(y, periods, True, True)
        for k, v in ta.items():
            print(f"  {k}: {v}")
    except Exception as exc:
        ta = None
        print(f"  tbats auto failed: {exc}")

    rf = ra = None
    if r_available():
        print("\n-- R forecast::tbats (fixed, via fitSpecificTBATS) --")
        try:
            rf = run_r_bench(y, "fixed", True, True, periods, k_vector)
            for kk, vv in rf.items():
                print(f"  {kk}: {vv}")
        except Exception as exc:
            rf = None
            print(f"  R fixed failed: {exc}")

        print("\n-- R forecast::tbats (auto-search) --")
        try:
            ra = run_r_bench(y, "auto", True, True, periods, k_vector)
            for kk, vv in ra.items():
                print(f"  {kk}: {vv}")
        except Exception as exc:
            ra = None
            print(f"  R auto failed: {exc}")
    else:
        print("\n-- R forecast::tbats: SKIPPED (Rscript not on PATH) --")

    print("\n-- Comparison --")
    print(f"  JAX (optimistix)       : {jo['wall_time']:.3f}s   SSR={jo['ssr']:.3f}   penalty={jo['penalty']:.2e}")
    print(f"  JAX (scipy L-BFGS-B)   : {js['wall_time']:.3f}s   SSR={js['ssr']:.3f}   penalty={js['penalty']:.2e}")
    if tf is not None:
        print(f"  Py tbats fixed         : {tf['wall_time']:.3f}s   SSR={tf['ssr']:.3f}")
    if rf is not None:
        print(f"  R  forecast fixed      : {rf['wall_time_s']:.3f}s   SSR={rf['ssr']:.3f}")
    if ta is not None:
        print(f"  Py tbats auto          : {ta['wall_time']:.3f}s   SSR={ta['ssr']:.3f}   k={ta['chosen_harmonics']}")
    if ra is not None:
        print(f"  R  forecast auto       : {ra['wall_time_s']:.3f}s   SSR={ra['ssr']:.3f}   k={ra['chosen_k']}")

    if rf is not None:
        print(f"\n  optimistix vs R fixed  : {rf['wall_time_s'] / max(jo['wall_time'], 1e-9):.2f}x")
    if ra is not None:
        print(f"  optimistix vs R auto   : {ra['wall_time_s'] / max(jo['wall_time'], 1e-9):.2f}x  (unfair: R searches k)")


if __name__ == "__main__":
    main()
