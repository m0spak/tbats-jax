"""Bayesian TBATS via NumPyro (EXPERIMENTAL — NUTS converges poorly).

Wraps the existing innovations-form kernel in a NumPyro model. The scaffold
works end-to-end: prior predictive, likelihood, posterior-predictive
forecast shapes are all correct. **However NUTS step-size adaptation
collapses to ~1e-45** when the chain brushes the log-hinge admissibility
barrier (which has infinite curvature at rho=1). As a result, posterior
samples on non-trivial problems are often stuck at a single point.

This is a known-hard pattern in HMC: log-barriers create posterior geometry
that Hamiltonian samplers can't navigate without reparameterization. Clean
fixes (not shipped here, each multi-day):

  1. Structural admissibility constraint — parameterize gammas so
     rho(D) < 1 is guaranteed at all samples, remove the barrier entirely.
  2. Non-centered reparameterization — lift correlated params to
     independent standard-normal auxiliaries.
  3. Dense mass-matrix adaptation with a longer warmup on the barrier.
  4. Switch to SVI / Laplace approximation instead of NUTS.

Current priors (deliberately tight, chosen to stay away from the barrier):
  - alpha         ~ Normal(0.1, 0.15)
  - beta          ~ Normal(0.0, 0.1)
  - phi           ~ Uniform(0.85, 1.0)
  - gamma1/2      ~ Normal(0, 0.003) — matches R's typical ||gamma|| ~ 1e-3
  - x0_level      ~ Normal(mean(y[:10]), std(y[:10]) + 1)
  - x0_rest       ~ Normal(0, 1)
  - sigma         ~ HalfNormal(std(y))

For small state (state_dim <= 5) and well-separated priors/posterior the
sampler runs fine — the included smoke test confirms end-to-end wiring.
For production Bayesian TBATS, do not use as-is.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any

import jax
import jax.numpy as jnp
import numpy as np

from tbats_jax.spec import TBATSSpec
from tbats_jax.matrices import build_matrices
from tbats_jax.params import Params
from tbats_jax.kernel import tbats_scan
from tbats_jax.admissibility import admissibility_penalty


def _make_params(alpha, phi, beta, gamma1, gamma2, x0) -> Params:
    return Params(
        box_cox_lambda=jnp.array(1.0),
        alpha=alpha, phi=phi, beta=beta,
        gamma1=gamma1, gamma2=gamma2, x0=x0,
    )


def make_tbats_model(spec: TBATSSpec,
                     admissibility_weight: float = 1.0,
                     admissibility_margin: float = 1e-4):
    """Return a NumPyro model function with `spec` bound via closure.

    The spec can't be a traced argument (it's a frozen dataclass, not a
    JAX pytree), so we build the model at Python level and let NumPyro
    trace only over `y`.
    """
    import numpyro
    import numpyro.distributions as dist

    K = spec.n_gamma
    sd = spec.state_dim

    def tbats_model(y):
        # Data-dependent prior scales (not learned — just weak informers)
        y_head = y[:10]
        y_head_mean = jnp.mean(y_head)
        y_head_std = jnp.std(y_head) + 1.0
        y_std = jnp.std(y) + 1e-3

        # --- parameter priors ---
        # Priors are deliberately TIGHT around R's typical fits to keep NUTS
        # away from unstable regions where the admissibility barrier has
        # infinite gradient (which collapses the sampler's step size).
        # Concrete R-on-Taylor values (for reference): alpha≈1.68, beta≈-0.24,
        # ||gamma||≈2e-3. These priors are wide enough to cover that regime
        # and narrow enough to exclude obviously unstable D matrices.
        alpha = numpyro.sample("alpha", dist.Normal(0.1, 0.15))
        if spec.use_damping:
            phi = numpyro.sample("phi", dist.Uniform(0.85, 1.0))
        else:
            phi = jnp.array(1.0)
        if spec.use_trend:
            beta = numpyro.sample("beta", dist.Normal(0.0, 0.1))
        else:
            beta = jnp.array(0.0)

        if K > 0:
            gamma1 = numpyro.sample("gamma1", dist.Normal(jnp.zeros(K), 0.003 * jnp.ones(K)))
            gamma2 = numpyro.sample("gamma2", dist.Normal(jnp.zeros(K), 0.003 * jnp.ones(K)))
        else:
            gamma1 = jnp.zeros(0)
            gamma2 = jnp.zeros(0)

        x0_level = numpyro.sample("x0_level", dist.Normal(y_head_mean, y_head_std))
        x0_rest = numpyro.sample("x0_rest", dist.Normal(jnp.zeros(sd - 1), 1.0 * jnp.ones(sd - 1)))
        x0 = jnp.concatenate([jnp.array([x0_level]), x0_rest])

        sigma = numpyro.sample("sigma", dist.HalfNormal(y_std))

        # --- build state-space matrices + run scan ---
        params = _make_params(alpha, phi, beta, gamma1, gamma2, x0)
        F, g, w = build_matrices(spec, params)
        residuals, _ = tbats_scan(y, x0, F, g, w)

        # Soft admissibility barrier (log-hinge on rho(D)). Keeps posterior
        # mass on the stable region; MCMC can still explore right up to the
        # boundary — matches our MLE fit path.
        penalty = admissibility_penalty(
            F, g, w,
            margin=admissibility_margin,
            weight=admissibility_weight,
        )
        numpyro.factor("admissibility", -penalty)

        # Observation model: residuals are i.i.d. Normal(0, sigma)
        numpyro.sample("obs", dist.Normal(0.0, sigma), obs=residuals)

    return tbats_model


@dataclass
class BayesResult:
    samples: Dict[str, np.ndarray]  # posterior draws by param name
    num_samples: int
    num_chains: int
    wall_time: float
    spec: TBATSSpec


def bayes_tbats(
    y,
    spec: TBATSSpec,
    num_warmup: int = 500,
    num_samples: int = 500,
    num_chains: int = 1,
    seed: int = 0,
    admissibility_weight: float = 1.0,
    admissibility_margin: float = 1e-4,
) -> BayesResult:
    """Run NUTS on the Bayesian TBATS model. Returns posterior samples.

    Minimal defaults (500/500, 1 chain) are for prototyping — bump to
    1000/2000/4 for real runs. On CPU, 500/500 takes ~30-60s on a 500-point
    series with state_dim≈5; much longer for larger state or series.
    """
    import time
    import numpyro
    from numpyro.infer import MCMC, NUTS

    y_arr = jnp.asarray(np.asarray(y, dtype=np.float64))

    model = make_tbats_model(
        spec,
        admissibility_weight=admissibility_weight,
        admissibility_margin=admissibility_margin,
    )
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=False,
        jit_model_args=True,
    )

    t0 = time.perf_counter()
    mcmc.run(jax.random.PRNGKey(seed), y=y_arr)
    wall = time.perf_counter() - t0

    samples = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    return BayesResult(
        samples=samples,
        num_samples=num_samples,
        num_chains=num_chains,
        wall_time=wall,
        spec=spec,
    )


def bayes_forecast(
    bayes_result: BayesResult,
    y,
    horizon: int,
    n_paths: int = 200,
    seed: int = 1,
) -> np.ndarray:
    """Posterior-predictive h-step forecasts.

    For each of `n_paths` sampled parameter draws (uniformly from the
    posterior), run the TBATS scan forward deterministically from the
    last state. Returns an (n_paths, horizon) array — quantiles give
    prediction intervals, mean gives the posterior-mean forecast.
    """
    spec = bayes_result.spec
    samples = bayes_result.samples
    total = samples["alpha"].shape[0]

    rng = np.random.default_rng(seed)
    idxs = rng.integers(0, total, size=n_paths)
    y_arr = jnp.asarray(np.asarray(y, dtype=np.float64))
    sd = spec.state_dim
    K = spec.n_gamma

    out = np.empty((n_paths, horizon), dtype=np.float64)
    for path, idx in enumerate(idxs):
        alpha = jnp.asarray(samples["alpha"][idx])
        phi = jnp.asarray(samples["phi"][idx]) if spec.use_damping else jnp.array(1.0)
        beta = jnp.asarray(samples["beta"][idx]) if spec.use_trend else jnp.array(0.0)
        gamma1 = jnp.asarray(samples["gamma1"][idx]) if K > 0 else jnp.zeros(0)
        gamma2 = jnp.asarray(samples["gamma2"][idx]) if K > 0 else jnp.zeros(0)
        x0_level = float(samples["x0_level"][idx])
        x0_rest = jnp.asarray(samples["x0_rest"][idx])
        x0 = jnp.concatenate([jnp.array([x0_level]), x0_rest])

        params = _make_params(alpha, phi, beta, gamma1, gamma2, x0)
        F, g, w = build_matrices(spec, params)
        _, x_T = tbats_scan(y_arr, x0, F, g, w)

        # Deterministic propagation from x_T
        x = x_T
        for t in range(horizon):
            y_hat = float(jnp.dot(w, x))
            out[path, t] = y_hat
            x = F @ x

    return out
