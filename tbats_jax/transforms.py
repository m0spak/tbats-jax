"""Smooth change-of-variables for parameters.

Mirrors R's forecast::tbats bounds as discovered in checkAdmissibility.R:
only `phi` has hard bounds [0.8, 1.0]; everything else is unbounded, with
the spectral-radius admissibility penalty acting as the sole constraint.

Over-constraining alpha/beta/gammas (as an earlier version did) severely
biased fits — R routinely lands at alpha > 1, beta < 0, with tiny gammas.

Layout mirrors params.py: [alpha, (phi), (beta), gamma1, gamma2, x0].
"""

import jax.numpy as jnp
import numpy as np

from tbats_jax.spec import TBATSSpec


PHI_LO, PHI_HI = 0.8, 1.0
# R: checkAdmissibility rejects lambda <= bc.lower=0 or >= bc.upper=1.
# Use a small epsilon to keep the sigmoid well away from the endpoints.
LAMBDA_LO, LAMBDA_HI = 1e-3, 1.0 - 1e-3


def _sigmoid(z):
    return 1.0 / (1.0 + jnp.exp(-z))


def _inv_sigmoid(p):
    p = jnp.clip(p, 1e-12, 1.0 - 1e-12)
    return jnp.log(p / (1.0 - p))


def _scaled_sigmoid(z, lo, hi):
    return lo + (hi - lo) * _sigmoid(z)


def _inv_scaled_sigmoid(x, lo, hi):
    p = (x - lo) / (hi - lo)
    return _inv_sigmoid(p)


def raw_to_natural(theta_raw: jnp.ndarray, spec: TBATSSpec) -> jnp.ndarray:
    """Map unconstrained theta_raw -> natural parameter vector.

    Only `lambda` (sigmoid into (0, 1)) and `phi` (sigmoid into [0.8, 1.0])
    are transformed; everything else is pass-through. Admissibility is
    enforced by the soft barrier in penalized_objective, not by transforms.
    """
    out = []
    i = 0
    if spec.use_box_cox:
        out.append(jnp.atleast_1d(_scaled_sigmoid(theta_raw[i], LAMBDA_LO, LAMBDA_HI))); i += 1
    out.append(jnp.atleast_1d(theta_raw[i])); i += 1                         # alpha
    if spec.use_damping:
        out.append(jnp.atleast_1d(_scaled_sigmoid(theta_raw[i], PHI_LO, PHI_HI))); i += 1
    if spec.use_trend:
        out.append(jnp.atleast_1d(theta_raw[i])); i += 1                     # beta
    K = spec.n_gamma
    out.append(theta_raw[i:i + K]); i += K                                    # gamma1
    out.append(theta_raw[i:i + K]); i += K                                    # gamma2
    out.append(theta_raw[i:i + spec.p]); i += spec.p                          # ar
    out.append(theta_raw[i:i + spec.q]); i += spec.q                          # ma
    out.append(theta_raw[i:i + spec.state_dim])                               # x0
    return jnp.concatenate(out)


def natural_to_raw(theta: np.ndarray, spec: TBATSSpec) -> np.ndarray:
    """Inverse of raw_to_natural."""
    out = []
    i = 0
    if spec.use_box_cox:
        out.append(float(_inv_scaled_sigmoid(jnp.asarray(theta[i]), LAMBDA_LO, LAMBDA_HI))); i += 1
    out.append(float(theta[i])); i += 1
    if spec.use_damping:
        out.append(float(_inv_scaled_sigmoid(jnp.asarray(theta[i]), PHI_LO, PHI_HI))); i += 1
    if spec.use_trend:
        out.append(float(theta[i])); i += 1
    K = spec.n_gamma
    out.extend(theta[i:i + K].tolist()); i += K
    out.extend(theta[i:i + K].tolist()); i += K
    out.extend(theta[i:i + spec.p].tolist()); i += spec.p
    out.extend(theta[i:i + spec.q].tolist()); i += spec.q
    out.extend(theta[i:i + spec.state_dim].tolist())
    return np.asarray(out, dtype=np.float64)
