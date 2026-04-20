# tbats-jax

JAX-native port of R's `forecast::tbats`. Innovations-form TBATS with
multi-seasonal Fourier harmonics, Box-Cox, missing data, and ARMA errors
— fit via `jax.grad` + optimistix BFGS on CPU or GPU. Ships with `vmap`
panel fitting that scales to thousands of series in a single call (10×
over single-core CPU at N=10000 on A100).

Experimental: scan-based Levenberg-Marquardt fit (`fit_lm`, TPU-compatible
but lower convergence quality than the main path) and a NumPyro Bayesian
scaffold (sampler converges poorly on non-trivial priors — see STATUS).

## Install

```bash
pip install tbats-jax                  # core (CPU/GPU)
pip install "tbats-jax[data]"          # + pyreadr for fetch_taylor (GPL-3 source, never bundled)
pip install "tbats-jax[bench]"         # + Python `tbats` for comparison
pip install "tbats-jax[bayes]"         # + numpyro (pins jax<0.10)
```

## Quickstart

```python
import jax
jax.config.update("jax_enable_x64", True)   # recommended for fit quality

import numpy as np
from tbats_jax import TBATSSpec, fit_jax, forecast
from tbats_jax.datasets import synthesize_daily

# Daily series with weekly + yearly seasonality (bundled synthetic generator)
y = synthesize_daily(n=730, seed=0)

spec = TBATSSpec(
    seasonal=((7.0, 3), (365.25, 5)),       # (period, k_harmonics) each
    use_trend=True,
    use_damping=True,
)

# Fit (~0.5s on CPU, ~0.02s on A100)
result = fit_jax(y, spec)

# 30-day forecast
preds = forecast(y, result.theta, spec, horizon=30)
```

### Batch 1000 series on GPU

```python
import numpy as np
from tbats_jax import TBATSSpec, fit_panel

panel = np.stack([make_series(s) for s in range(1000)])  # shape (1000, T)
spec  = TBATSSpec(seasonal=((24.0, 3), (168.0, 5)), use_trend=True, use_damping=True)

thetas_raw, nlls, compile_t, wall_t = fit_panel(panel, spec)
# A100 @ T=1500, N=1000: ~25s wall for all 1000 fits
```

### Auto-select seasonal harmonics

```python
from tbats_jax import auto_fit_jax_cv

r = auto_fit_jax_cv(y, periods=(7.0, 365.25),
                   use_trend=True, use_damping=True,
                   val_size=30)  # walk-forward CV over last 30 points
# r.spec has the chosen k-vector; r.fit.theta is the final fit
```

## Scope (v0.1.0)
- trend (optional), damping (optional), multi-seasonal harmonics
- Box-Cox transform (optional, `use_box_cox=True`)
- Missing-data support (NaN values in `y` handled automatically)
- innovations state-space form: `y_hat_t = w' x_{t-1}`, `x_t = F x_{t-1} + g e_t`
- Gaussian negative log-likelihood + hybrid admissibility barrier (exact
  eigvals on CPU, power-iteration spectral norm on GPU/TPU) + gamma-ridge
- Two optimizers: scipy L-BFGS-B (`fit`) and optimistix BFGS (`fit_jax`)
- Uniform panel fit via `vmap` (`fit_panel`)
- Heterogeneous panel fit across different specs (`fit_panel_hetero`)
- Auto k-vector search (`auto_fit_jax`) — R-compatible greedy per-period AIC
- CPU or GPU (JAX device-agnostic); Colab notebook in `notebooks/`
- ARMA errors (`p`, `q` in TBATSSpec)
- TPU-viable scan-based LM optimizer (`fit_lm`, `fit_lm_multistart`)
- Experimental Bayesian TBATS via NumPyro (HMC)

## Layout

```
tbats_jax/
├── tbats_jax/          # library (pure, device-agnostic)
│   ├── spec.py         # TBATSSpec
│   ├── params.py       # pack / unpack / init
│   ├── matrices.py     # F, g, w builders
│   ├── kernel.py       # lax.scan + likelihood
│   ├── forecast.py     # h-step point forecast
│   └── fit.py          # scipy L-BFGS-B + jax.grad
├── benchmarks/
│   ├── data.py
│   ├── bench_single.py # JAX vs Python `tbats` package
│   └── bench_panel_full.py  # vmap vs R sequential
├── tests/test_smoke.py
└── requirements.txt
```

## Setup

```sh
cd tbats_jax
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Run

```sh
pytest tests/                        # smoke tests
python -m benchmarks.bench_single    # single-series SSR, 4 backends
python -m benchmarks.bench_oos       # out-of-sample MAE/RMSE, 4 backends
python -m benchmarks.bench_panel_full [N] [T]  # full fits: vmap vs R sequential
python -m benchmarks.colab_panel_gpu [N] [T]   # device-agnostic; GPU on Colab
# Real data (requires R + one-time fetch):
Rscript benchmarks/fetch_real_data.R data
python -m benchmarks.bench_real      # forecast::taylor OOS
```

The R comparison is auto-detected. If `Rscript` is on `PATH` and the
`forecast` package is installed, `bench_single` includes it; otherwise it
cleanly skips. To enable:

```sh
brew install r
Rscript -e 'install.packages("forecast", repos="https://cloud.r-project.org")'
```

## Colab

Same code. Switch runtime → GPU, install requirements, run the same
benchmarks. No source changes required. Enable float64 in the first cell:

```python
import jax; jax.config.update("jax_enable_x64", True)
```

## Results on Apple Silicon (M-series CPU, float64)

Series: n=1500, two seasonal periods (24, 168), k=(3,5), trend+damping.
Four-way comparison: JAX vs Python `tbats` (pure numpy port) vs R
`forecast::tbats` (the original, C++ kernel via Rcpp/Armadillo).

### Single-series in-sample fit (SSR) — `bench_single.py`

Synthetic series, n=1500, periods=(24, 168), k=(3, 5), trend+damping.

| Implementation                          | Wall    | SSR       | Notes                                |
|-----------------------------------------|---------|-----------|--------------------------------------|
| **JAX `fit_jax` (optimistix BFGS)**     | 0.72 s  | **361.8** | on-graph, + gamma-ridge + log-hinge  |
| R `forecast:::fitSpecificTBATS` (same k) | **0.35 s** | 374.9 | Original C++ kernel, Nelder-Mead     |
| JAX `fit` (scipy L-BFGS-B)              | 0.80 s  | 379.9     | scipy path retained for comparison   |
| Python `tbats` (same k pinned)          | 1.19 s  | 375.0     | Nelder-Mead over pure numpy          |
| R `forecast::tbats` (auto-search)       | 1.57 s  | 369.5     | chose k=[6,6]                        |
| Python `tbats` (auto-search)            | 11.5 s  | 373.0     | chose k=[6,2]                        |

**JAX finds better in-sample SSR than R** (361.8 vs 374.9 at matched
structure) at ~2× R's wall time. The C++ baseline is hard to beat on a
single fit, but quality is now unambiguously at parity.

### Panel fit — `bench_panel_full.py`

32 synthetic series × 1500 obs each. `fit_panel` JIT-compiles one fused
kernel across all series; R runs sequential `Rscript` calls.

| Backend                  | Total    | Per-series | Mean SSR  |
|--------------------------|----------|------------|-----------|
| **JAX `fit_panel` (vmap)** | **7.3 s** | **227 ms** | **374.8** |
| JAX loop (no vmap)       | 59.3 s   | 1852 ms    | 375.2     |
| R forecast sequential    | 23.7 s   | 740 ms     | 382.8     |

**JAX vmap is 3.3× faster than R sequential AND finds lower mean SSR
(374.8 vs 382.8, 2.1% better).** This is the headline result: parity quality
at scaled throughput. At N=100 per-series time stays at ~213 ms as JIT
overhead amortizes further.

### Out-of-sample forecast accuracy — `bench_oos.py`

Train on first 1350 points, forecast last 150.

| Implementation        | Wall    | Train SSR | Test MAE   | Test RMSE  |
|-----------------------|---------|-----------|------------|------------|
| R `forecast` (fixed k) | 0.32 s | 334       | **0.4273** | **0.5250** |
| R `forecast` (auto)   | 1.51 s  | 330       | 0.4224     | 0.5193     |
| Python `tbats` (auto) | 10.5 s  | —         | 0.4280     | 0.5260     |
| JAX `fit_jax`         | 0.47 s  | 335       | 0.4447     | 0.5384     |

**Test MAE gap vs R: 4.1%**, down from ~14% before the v0.0.3 regularization
fixes. JAX now also beats Python `tbats` on wall time by 22×.

### Real dataset — `bench_real.py` (`forecast::taylor`)

Half-hourly UK electricity demand, n=4032, periods=(48, 336), k=(3, 5).
One-week hold-out (336 points). The canonical TBATS benchmark.

| Backend                    | Wall     | Train SSR | Test MAE   | Test RMSE  |
|----------------------------|----------|-----------|------------|------------|
| **R forecast fixed**       | **0.33 s** | 6.99e8  | **1030**   | **1332**   |
| JAX `fit_jax`              | 4.16 s   | 6.96e8    | **1041**   | 1343       |
| R forecast auto (k=[12,5]) | 10.03 s  | 3.01e8    | 1273       | 1499       |

**JAX test MAE is within 1.1% of R's fixed fit** — essentially parity on the
metric that matters. The wall-time penalty is larger than on short series
(4.2 s vs 0.3 s) because n=4032 makes the scan longer; this scales with
panel width, not `vmap`'d series count.

## Accelerator comparison — `fit_panel`

Two workloads tested: a daily panel (T=730, weekly+yearly seasonality) for
a baseline sanity check, and an hourly panel (T=1500, periods=24+168) which
is where the GPU story actually matters for long-series production data.

### Daily panel (T=730, k=(3,5))

| Backend          | Compile | Warm wall | Per-series | vs CPU | Notes |
|------------------|---------|-----------|------------|--------|-------|
| **Apple Silicon M-series CPU** | 5.4 s | 4.7 s (N=100) | **45 ms** | 1.0× | eigvals path, exact rho |
| **CUDA T4 16 GB**              | 33 s  | 24 s (N=1000) | **24 ms** | **1.85×** | power-iter path |
| **CUDA T4 16 GB** (N=2000)     | 62 s  | 56 s | 28 ms | 1.60× | per-series creeps as HBM loads up |
| **TPU v5e-1**                  | 1459 s | 1429 s (N=1000) | 1429 ms | **0.03× (slower!)** | see below |

### Hourly panel (T=1500, k=(3,5)) — the headline

Per-series CPU work is 4× heavier here (196 ms vs 45 ms). T4 hit its
bandwidth wall and lost to CPU at N=5000. A100 had the headroom to scale:

| Backend | N=1000 | N=5000 | N=10000 | N=20000 |
|---|---|---|---|---|
| **CPU** (Apple Silicon M-series) | 196 ms | 196 ms* | 196 ms* | 196 ms* |
| **CUDA T4 16 GB** | 104 ms (1.88×) | **214 ms (0.92× — lost)** | — | — |
| **CUDA A100 40 GB** | **53 ms** | **20 ms** | **20 ms** | 30 ms |
| vs CPU | 3.7× | **9.9×** | **9.8×** | 6.6× |
| vs R `forecast::tbats` sequential | 6.5× | **17.3×** | **17.1×** | 11.5× |

<sub>*CPU at large N is linear-extrapolation from the measured 196 ms/series at N=500.</sub>

**A100 fits 10,000 independent TBATS models in 200 seconds.** Sequential
R `forecast::tbats` on the same workload would take ~57 minutes. CPU
(single core) takes ~33 minutes. The GPU sweet spot is N=5000–10000 where
per-series time drops 2.7× from the N=1000 launch-bound regime to the
steady-state ~20 ms.

**Why A100 scales where T4 didn't:**
- **5× HBM bandwidth** (1.6 TB/s vs 320 GB/s) — T4's bandwidth wall hit at
  N=5000 simply doesn't appear until N>20000 on A100
- **Enough SMs to saturate on the vmap axis** — at N=1000 both T4 and A100
  are launch-bound, but A100 scales the lanes per kernel; T4 stalls
- **39× FP64 throughput** — mostly unused in a launch-bound kernel, but
  helps at the margins

### Why TPU v5e-1 loses so badly

The compile alone is 24 min and the warm fit is another 24 min on a
workload GPU finishes in under a minute. Three structural reasons:

1. **`optimistix.BFGS` uses `lax.while_loop`.** TPU XLA compilation of
   unbounded-iteration loops is expensive. Every step size, line search,
   and convergence check is a symbolic branch the compiler has to lower
   into TPU bundles. GPU XLA handles this gracefully; TPU XLA does not.
2. **v5e-1 is a single "Lite" chip**, optimized for inference matmul
   throughput (bf16/int8). Our per-step FLOPs are tiny (~300 on an 18×18
   matmul) — well below the granularity the TPU wants for good
   utilization.
3. **Scan serialization over T=730** hits the same launch-coordination
   issue on TPU that it hits on GPU — but without GPU's faster step
   dispatch.

A fixed-iteration optimizer (e.g., a hand-written BFGS expressed as a
`lax.scan` with bounded steps, instead of `while_loop`) would likely fix
this and make v5e-1 competitive. That's a real rewrite, not a tuning
change — we'd give up adaptive convergence tolerance in exchange for
TPU-friendly compilation.

### Practical recommendations

- **For development and small panels (N ≤ 200):** use CPU. `eigvals`
  gives tight barriers, compile is fast, per-series throughput is fine.
- **For moderate panels (N = 500–2000):** T4 (Colab free tier) is enough.
  Expect **~1.7-2× over CPU**, compile ~30 s.
- **For large hourly panels (N = 1000–10000, T ≈ 1500):** use **A100**
  (Colab Pro). **Scales to 10× over CPU, 17× over R sequential** at
  N=5000–10000. Compile 100–200 s, amortizes immediately.
- **For very large panels (N > 20000):** compile cost and HBM load start
  to degrade per-series time on A100 too. Either chunk the panel, or use
  H100 if available (untested here).
- **Don't use v5e-1 TPU without a fixed-iteration optimizer rewrite.**
  Its design assumptions don't match our `while_loop`-based kernel.

## How the v0.0.2 → v0.0.3 regressions were fixed

The earlier OOS gap (14% synthetic, 2× on taylor) traced to three things,
each discovered by reading `forecast::tbats` source and diffing fitted
parameters backend-to-backend:

1. **Over-constrained transforms.** v0.0.2 sigmoid-mapped alpha to [0,1],
   beta to [0,1], gammas to [-0.5, 0.5]. The R reference (`checkAdmissibility.R`)
   in fact only bounds `phi ∈ [0.8, 1.0]`; alpha, beta, and gammas are
   unbounded. R's fitted `alpha` on taylor is 1.68 (!) and `beta` is −0.24.
   Fix: identity transforms for everything except phi.

2. **Wrong starting values.** v0.0.2 used `gammas=0.001, phi=0.98`. R uses
   `gammas=0, phi=0.999` (fitTBATS.R lines 160-179). Matching gave another
   ~9% MAE reduction on taylor.

3. **Missing implicit regularization.** R's `makeParscale()` sets
   `parscale=1e-5` for gammas, which makes Nelder-Mead take microscopic
   steps in gamma space — an accidental regularizer that keeps ||gamma||
   tiny (~1e-3 in R vs ~2.9 in our v0.0.2). We match this explicitly with
   an L2 ridge on gammas, `gamma_ridge=1e6` by default. This closed most
   of the remaining taylor gap.

## What's honest to claim today

**Works well:**
- Single-series fit quality better than R's Nelder-Mead on synthetic data.
- Panel `vmap` is 3.3× faster than R sequential *and* 2% better mean SSR.
- Out-of-sample: within 4% of R on synthetic, within 1% on real taylor.
- Faithful mirror of forecast::tbats bounds, init, and implicit regularization.

**Remaining gaps (v0.0.6):**
- Single-series wall time ~2× R (compiled C++ vs JIT'd JAX). Won't close
  without dropping past JAX, not worth it.
- Box-Cox is implemented (`use_box_cox=True`) but interacts with
  `gamma_ridge` defaults: turn ridge down to ~1e3-1e4 when Box-Cox is on
  until auto-scaling is added.
- No ARMA errors.
- Auto k-search is implemented but AIC selection doesn't always predict
  OOS performance on specific datasets — same issue R has.
- `jax-metal` not a practical backend; CPU + CUDA only.
- **TPU v5e-1 not viable** with current BFGS-while-loop design. Fixed-
  iteration optimizer required for TPU compile to be reasonable.

**The structural value proposition** — panel `vmap`, gradients for Bayesian
TBATS, differentiable TBATS as a layer — is now backed by quality parity
with the R reference, not just "works in JAX."

Panel micro-bench (50 series × 1000 obs):

| Mode                               | Total wall | Per-series |
|------------------------------------|------------|------------|
| Looped JAX fits                    | 29.5 s     | 590 ms     |
| Batched likelihood (vmap, no fit)  | 1.4 ms     | 0.03 ms    |

The batched likelihood result is the headline: once the optimizer moves
inside JAX (jaxopt/optimistix), the full fit becomes one fused vmapped
call — the path to real panel scaling.

## Known limitations (roadmap)

- No Box-Cox transform yet (add lambda parameter + Jacobian term to NLL)
- No ARMA errors (add AR/MA state extension + coefficient parameters)
- Seed state x0 initialized naively; port OLS warmup from fitTBATS.R
- Admissibility via soft penalty; could tighten using differentiable
  reparameterization to guaranteed-stable region
- No auto-search over k-vector yet (outer discrete loop belongs in Python)
- vmap over panels is only demonstrated for likelihood eval; full fit
  batching requires a JAX-native optimizer replacing scipy.optimize
