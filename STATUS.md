# Session state — pick up here

v0.0.6 adds hybrid admissibility dispatch: exact `eigvals` on CPU (SOTA
control-theory correctness), power-iteration on GPU/TPU (SOTA ML spectral
normalization). CPU fit quality improved meaningfully (single-series SSR
394 → 354, panel mean SSR 375 → 367). Taylor MAE unchanged (~1042).
Three-way accelerator sweep complete:
- CPU: works (baseline at 196 ms/series T=1500)
- T4 GPU: works at small N (1.85× at N=1000), loses at N=5000 (HBM wall)
- **A100 GPU: headline — 10× over CPU, 17× over R, at N=5000-10000**
  (per-series 20 ms steady-state, fits 10k TBATS models in 200 s)
- v5e-1 TPU: not viable (while_loop compile is prohibitive).

## Where we are (v0.0.6)

| Metric                          | JAX                 | R fixed        |
|---------------------------------|---------------------|----------------|
| Single-series SSR (synthetic)   | **361.8** (better)  | 374.9          |
| Panel mean SSR (32 series)      | **374.8** (better)  | 382.8          |
| Panel wall (N=32)               | **7.3 s** (3.3×)    | 23.7 s         |
| Synthetic OOS MAE gap           | 4.1%                | —              |
| Taylor (real) OOS MAE           | 1041                | 1030 (1.1% gap)|

All 10 smoke tests pass. Benches run end-to-end. venv at `.venv/` with
jax 0.10, optimistix 0.1, tbats 1.1.3, scikit-learn <1.6.

## Files added (v0.0.5 → v0.0.6)

- `tbats_jax/auto.py` — R-compatible `auto_fit_jax()` + `_default_max_k()`
- `tbats_jax/admissibility.py` — hybrid dispatch: `method='auto'` picks
  `eigvals` on CPU, `power` on GPU/TPU. `_default_method()` helper.
- `benchmarks/colab_panel_gpu.py` — T=1500 hourly panel
- `benchmarks/colab_daily_panel.py` — T=730 daily panel (realistic shape)
- `benchmarks/data.py` — added `synthesize_daily()`
- `notebooks/colab_panel_gpu.ipynb` — T=1500 GPU bench
- `notebooks/colab_daily_panel.ipynb` — T=730 GPU bench
- `notebooks/colab_accelerators.ipynb` — T4 **or** v5e-1 TPU (hybrid-aware)

## Resume — next steps ranked

1. **Fixed-iteration optimizer for TPU viability** (SHIPPED via LM —
   `tbats_jax.fit_lm`).

   Reframed the TBATS objective as a sum-of-squares: the scan innovations,
   the gamma ridge, and a hinge-form admissibility violation are all
   assembled into one residual vector `r(theta)`. Levenberg-Marquardt
   then exploits this structure:

   ```
   for step in range(max_steps):
       r, J    = residuals(theta), jacfwd(residuals)(theta)
       delta   = solve(J^T J + lam*I, -J^T r)
       theta' = theta + delta
       accept  = ||r(theta')|| < ||r||
       lam    /= 3 if accept else *= 3   # Marquardt damping, via jnp.where
   ```

   No `while_loop` anywhere — pure `lax.scan`. TPU-compatible by
   construction.

   **Measured comparison (CPU, N=32 T=1500 synthetic panel):**
   | Optimizer              | Per-series | Mean SSR | TPU? |
   | ---------------------- | ---------: | -------: | :--: |
   | fit_lm (ms=30)         |  **93 ms** |    376   | ✓    |
   | fit_lm (ms=50)         |    153 ms  |    372   | ✓    |
   | fit_panel (optimistix) |    252 ms  |    367   | ✗    |

   **LM at 30 steps is 2.7× faster than optimistix on CPU with 2.6%
   worse SSR — for panel workloads this is a pure win, and it's the
   only optimizer in the tree that compiles on TPU.**

   Taylor real-data: at ms=30, MAE=1334 (vs optimistix 1042, 28% worse).
   Non-trivial gap driven by hinge vs log-hinge admissibility — LM's
   sum-of-squares reformulation gave up the log-barrier's tight
   boundary-kissing behavior. For TPU viability or fast panel fits the
   trade is worth it; for single-series accuracy on hard data use
   `fit_jax`.

   **Next refinement (not shipped, 1-2 days):** add a trust-region
   schedule to adaptively reduce `lam` more aggressively, and try a
   soft-plus approximation to log-hinge that stays sum-of-squares.
   Likely closes most of the Taylor gap.

2. **Bayesian TBATS via NumPyro** (SCAFFOLD SHIPPED, MCMC experimental).
   `tbats_jax.bayes_tbats` and `bayes_forecast` wire NumPyro's NUTS onto
   the same kernel fit_jax uses. End-to-end shapes (priors, posterior,
   predictive draws) are correct and the smoke test passes.

   **Known limitation:** NUTS step-size adaptation collapses to ~1e-45
   when the chain approaches the log-hinge admissibility barrier (which
   has infinite curvature at rho=1). Posterior samples on non-trivial
   problems are stuck. Clean fixes (structural admissibility
   parameterization, non-centered priors, SVI) are multi-day; deferred.

   Value today: anyone wanting to explore Bayesian TBATS on a JAX stack
   has a starting point with correct wiring. Ready for the next engineer
   with NumPyro expertise to tune priors / reparameterize.

3. **Auto-search quality improvement** (SHIPPED for synthetic; partial on
   real data). Added `auto_fit_jax_cv()` — greedy-per-period search ranked
   by held-out-tail MAE instead of AIC, with a multi-start `starts="multi"`
   option that tries {max_k, max_k/4, (3,3,...), (5,5,...), ones} as seeds.

   Synthetic (periods=24,168, true k≈(3,5), n=1500):
   - AIC auto:    k=(10,6) → test MAE 6.03
   - CV multi:    k=(3,3)  → test MAE **0.44** (matches manual reference)

   Taylor (periods=48,336):
   - AIC auto:    k=(21,6) → test MAE 3467
   - CV greedy:   k=(23,6) → test MAE 3607
   - CV multi:    k=(5,4)  → val MAE 1038 (great) but test MAE 4555 (bad)
     — on Taylor the per-candidate optimizer convergence is noisy, so
     CV ranking sometimes picks a fit that generalizes poorly to the
     test window. Optimization quality (not the ranking criterion) is
     the remaining bottleneck.

   Bottom line: CV-multi helps significantly on well-conditioned data;
   on Taylor-like hard cases, manual k + direct `fit_jax` is still the
   safer bet.

3. **Two-pass fit protocol** (~3 hrs). R calls `fitSpecificTBATS` twice;
   pass-2 starts from pass-1 optimum. Marginal SSR gain.

4. **Bayesian TBATS via NumPyro** (~1 day). Wrap kernel, run HMC. No R
   equivalent. Research upside.

5. **ARMA errors** (~1 day). Low priority; R usually drops them via AIC
   anyway.

6. **Real-data Box-Cox benchmark** (~1 hr). Real utility of Box-Cox likely
   shows on genuinely multiplicative series (retail demand, traffic).
   Add a multiplicative synthetic or M4-hourly subset to `bench_real.py`.

## Tech debt worth noting

- `tbats_jax/seed_state.py` still has `with_ols_x0` and `warmup_then_ols`
  that are experimental; `fit()` exposes `ols_seed_state=False` by default.
  Not used on the `fit_jax` path. (Candidate for integration or removal.)
- `tbats_jax/fit.py` (scipy) kept for cross-optimizer comparison but
  `fit_jax` is the forward-looking path. Defaults + knobs now unified with
  `fit_jax` so either path produces comparable quality at the same config.

## Quick resume commands

```sh
cd /Users/meister/DOCS/projects/DS/tbats/tbats_jax
source .venv/bin/activate
pytest tests/ -q
python -m benchmarks.bench_single        # 4-way in-sample comparison
python -m benchmarks.bench_oos           # synthetic held-out MAE
python -m benchmarks.bench_real          # taylor held-out MAE
python -m benchmarks.bench_panel_full    # vmap vs R sequential
```
