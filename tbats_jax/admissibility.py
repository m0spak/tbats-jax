"""Soft admissibility penalty with hybrid backend dispatch.

TBATS is stable iff the spectral radius rho(D) < 1 where D = F - g w^T.
We penalize rho only, not the full eigenvalue vector — that's what
matters for stability and also matches modern ML practice (spectral
normalization, Miyato et al. 2018).

Hybrid dispatch:
  - CPU : exact rho via `jnp.linalg.eigvals(D)` + max|.|. LAPACK on 18x18
          is microseconds, and gives the true spectral radius (tighter
          barrier than the spectral-norm upper bound).
  - GPU : spectral-norm upper bound via power iteration on D^T D. Avoids
          cuSolver's high launch overhead on tiny eigendecompositions.
          rho(D) <= ||D||_2, so the barrier is conservative but valid.
  - TPU : same as GPU (power iter). `jnp.linalg.eigvals` on TPU falls back
          to CPU and is slower than the scan.

You can force a specific rho method via `method=` ("eigvals", "power",
or "auto"). Defaults to "auto" which picks per backend.
"""

import jax
import jax.numpy as jnp


def _spectral_radius_power_iter(D, n_iter: int = 10):
    """Power iteration on D^T D → returns ||D||_2 (upper bound on rho(D)).

    For non-normal matrices (TBATS D is always non-normal) this is strictly
    larger than the true spectral radius, which makes the barrier conservative.
    """
    d = D.shape[0]
    v = jnp.ones(d) / jnp.sqrt(d)

    def step(v, _):
        v = D @ v
        v = D.T @ v
        v = v / (jnp.linalg.norm(v) + 1e-30)
        return v, None

    v, _ = jax.lax.scan(step, v, None, length=n_iter)
    Dv = D @ v
    DtDv = D.T @ Dv
    sigma_sq = jnp.dot(v, DtDv)
    return jnp.sqrt(jnp.maximum(sigma_sq, 0.0))


def _spectral_radius_eigvals(D):
    """Exact spectral radius via eigendecomposition. Fast on CPU (LAPACK)
    but has launch-overhead pitfalls on GPU/TPU for tiny matrices."""
    eigs = jnp.linalg.eigvals(D)
    return jnp.max(jnp.abs(eigs))


def _default_method() -> str:
    """Pick the rho method based on the default JAX backend at trace time."""
    try:
        backend = jax.default_backend()
    except Exception:
        backend = "cpu"
    return "eigvals" if backend == "cpu" else "power"


def admissibility_penalty(F, g, w,
                          margin: float = 1e-3,
                          weight: float = 1e4,
                          form: str = "log_hinge",
                          method: str = "auto",
                          n_power_iter: int = 10):
    """Soft admissibility barrier on rho(D) where D = F - g w^T.

    `method` in {'auto', 'eigvals', 'power'}:
      - auto   : eigvals on CPU, power iteration elsewhere (default)
      - eigvals: always use jnp.linalg.eigvals + max|.| (exact rho)
      - power  : always use power iteration on D^T D (upper bound ||D||_2)

    `form` in {'hinge', 'log', 'log_hinge'}:
      - log_hinge (default): log barrier inside, hinge quadratic outside
    """
    D = F - jnp.outer(g, w)

    chosen = _default_method() if method == "auto" else method
    if chosen == "eigvals":
        rho = _spectral_radius_eigvals(D)
    elif chosen == "power":
        rho = _spectral_radius_power_iter(D, n_iter=n_power_iter)
    else:
        raise ValueError(f"unknown method {method!r}")

    if form == "hinge":
        violation = jnp.maximum(rho - (1.0 - margin), 0.0)
        return weight * violation ** 2

    if form == "log":
        slack = jnp.maximum(1.0 - rho, margin)
        return -weight * jnp.log(slack)

    # log_hinge
    slack = 1.0 - rho
    w_hinge = 1e4 * weight
    inside = -weight * jnp.log(jnp.maximum(slack, margin))
    over = jnp.maximum(margin - slack, 0.0)
    outside = w_hinge * over ** 2
    return inside + outside
