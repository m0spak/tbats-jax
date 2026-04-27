# Developer notes

Living document — current state, measured numbers, known limitations,
and ranked next-steps queue. User-facing docs live in `README.md` and
the API docstrings. This file is the resumption anchor for any
contributor picking up work on this repo.

**Current release:** v0.1.0 shipped to PyPI as `tbats-jax`. 24 tests
passing. Codeberg primary, GitHub mirror live. Working tree clean.

## Published artifact

```
PyPI      : https://pypi.org/project/tbats-jax/0.1.0/
Codeberg  : https://codeberg.org/mospak/tbats-jax     (origin)
GitHub    : https://github.com/m0spak/tbats-jax       (mirror, synced by Codeberg)
Install   : pip install tbats-jax                     (3.10–3.14, all tested)
Extras    : [data] pyreadr, [bench] tbats+matplotlib, [bayes] numpyro, [dev] pytest
```

## Where we are (v0.1.0)

| Metric | JAX (ours) | R reference | Status |
|---|---|---|---|
| Single-series SSR (synthetic) | 361.8 | 374.9 | JAX 3.5% better |
| Taylor test MAE (n=3696) | 1042 | 1030 | 1.2% gap |
| Panel mean SSR (N=32) | 374.8 | 382.8 | JAX 2.1% better |
| Panel wall vs R (N=32) | 7.3 s | 23.8 s | **3.3×** faster |
| A100 N=10000 warm wall | 200 s | 57 min (R) | **17× warm / 8.5× cold** |

All numbers reproduced on the PyPI install in Colab Pro — matches internal
dev measurements bit-for-bit.

## Shipped features

- TBATSSpec with multi-seasonal Fourier, Box-Cox, ARMA(p,q), trend+damping
- `fit`, `fit_jax` (optimistix BFGS), `fit_lm` (Levenberg-Marquardt,
  TPU-compatible), `fit_panel` (vmap), `fit_panel_hetero` (bucketed),
  `fit_panel_lm`, `fit_lm_multistart`
- `auto_fit_jax` (AIC-ranked), `auto_fit_jax_cv` (CV-ranked with multi-start)
- `bayes_tbats`, `bayes_forecast` — experimental, NUTS scaffold
- `forecast` — h-step point forecast
- `tbats_jax.datasets` — synthesize_daily, synthesize_two_seasonal,
  fetch_taylor (never bundled)
- Hybrid admissibility (exact eigvals on CPU, power iter on GPU/TPU)
- Missing data via NaN-masked scan (opt-in, no cost when no NaNs)

## Commits on main

```
ab8b748  docs: split A100 benchmarks into warm vs cold
2e82f35  docs: PyPI / Python / License / Source badges
390fbe5  v0.1.0: ARMA errors + PyPI packaging   ← tagged v0.1.0
f56b5df  fit_lm_multistart
c7db800  fit_lm adam_steps warmup kwarg
5d9b03f  STATUS: fit_lm Taylor diagnostic
f35fbfa  fit_lm: Levenberg-Marquardt (scan-based)
f3dbc9f  bayesian.py: NumPyro scaffold (experimental)
f9f1c15  auto_fit_jax_cv: CV + multi-start
4c14191  v0.0.7: JAX TBATS baseline               ← tagged v0.0.7
```

## Tech debt / known limitations (NOT blockers)

1. **Bayesian NUTS step-size collapses** to ~1e-45 on non-trivial priors.
   `bayesian.py` is scaffold-only; known issue documented. Real fix needs
   structural admissibility reparameterization or switch to SVI.

2. **Auto k-search is noisy on real data.** AIC picks wrong on Taylor;
   CV+multi-start helps on synthetic but not consistently on hard cases.
   Working as designed; problem is the TBATS loss landscape itself.

3. **OLS init is asymmetric across optimizers.** `fit_lm` defaults to
   `init_method='ols'` and benefits dramatically (~60% MAE reduction on
   Taylor cold-start). `fit_jax` (BFGS) does NOT — empirically OLS init
   makes BFGS worse on Taylor (1042 → 1499). Don't reflexively apply OLS
   to fit_jax; the asymmetry is real (LM converges within a basin, BFGS
   does basin-finding too — they need different starts). Diagnostic:
   `python -m benchmarks.diag_init_quality`.

## v0.1.1 changelog (unreleased)

- **JIT cache lift in `fit_jax` and `fit_panel`** (commit `5a254b7`):
  module-level `lru_cache` factories so repeat calls with the same
  spec/shape skip recompile. Saves ~200 s per cache hit on Colab A100.
- **OLS seed-state init by default for `fit_lm` / `fit_lm_multistart`**
  (this commit): closes Taylor cold-start MAE from ~2800 to ~1120.
  Mirrors `fitTBATS.R:327-368`. Opt out via `init_method='naive'`.

## Resume — next steps ranked

1. **Bayesian reparameterization** (multi-day). Replace the log-hinge
   admissibility barrier with a structural parameterization that
   guarantees `ρ(D) < 1`. Unblocks NUTS sampling on real problems.

2. **PyPI housekeeping** (10 min): narrow the upload token scope to
   project-only at <https://pypi.org/manage/account/token/>; set up
   `~/.pypirc` for no-prompt future releases.

3. **Cut v0.1.1** (~30 min): bump pyproject to 0.1.1, build, TestPyPI
   smoke test, real PyPI upload. Two real fixes already bundled (JIT
   cache + OLS init). Tag `v0.1.1` after PyPI publish.

## Quick resume commands

```sh
cd /Users/meister/DOCS/projects/DS/tbats/tbats_jax
source .venv/bin/activate
pytest tests/ -q                             # 24 tests
python -m benchmarks.bench_single            # 4-way in-sample
python -m benchmarks.bench_real              # Taylor OOS (requires R)
python -m benchmarks.bench_panel_full        # vmap vs R sequential

# Build + publish a future version (after bumping pyproject.toml version):
rm -rf dist && python -m build && twine check dist/*
twine upload --repository testpypi dist/*   # dry-run
twine upload dist/*                          # real
```
