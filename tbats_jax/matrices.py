"""Build TBATS state-space matrices F, g, w from (spec, params).

State layout: [level, (slope), (s1_cos, s1_sin, s2_cos, s2_sin, ...)]

Transition F:
  level row:  [1, phi, 0*, 0*, ...]
  slope row:  [0, phi, 0*, 0*, ...]
  seasonal blocks: 2x2 rotation by lambda_j = 2*pi*j/m (per (m,k), per j=1..k)

Observation w (y_hat = w @ x_{t-1}):
  [1, phi, 1, 0, 1, 0, ...]  (only cos components contribute to obs)

Innovation g (x_t = F @ x_{t-1} + g * e_t):
  [alpha, beta, gamma1_1, gamma2_1, gamma1_2, gamma2_2, ...]
"""

import jax.numpy as jnp
import numpy as np

from tbats_jax.spec import TBATSSpec
from tbats_jax.params import Params


def _seasonal_lambdas(spec: TBATSSpec) -> np.ndarray:
    """Flat array of angular frequencies for every harmonic."""
    out = []
    for m, k in spec.seasonal:
        for j in range(1, k + 1):
            out.append(2.0 * np.pi * j / m)
    return np.asarray(out, dtype=np.float64)


def make_F(spec: TBATSSpec, phi: jnp.ndarray) -> jnp.ndarray:
    d = spec.state_dim
    F = jnp.zeros((d, d))

    F = F.at[0, 0].set(1.0)
    row_level = 0

    if spec.use_trend:
        F = F.at[0, 1].set(phi)
        F = F.at[1, 1].set(phi)
        start_seasonal = 2
    else:
        start_seasonal = 1

    lambdas = _seasonal_lambdas(spec)
    for i, lam in enumerate(lambdas):
        pos = start_seasonal + 2 * i
        c = jnp.cos(lam)
        s = jnp.sin(lam)
        F = F.at[pos, pos].set(c)
        F = F.at[pos, pos + 1].set(s)
        F = F.at[pos + 1, pos].set(-s)
        F = F.at[pos + 1, pos + 1].set(c)

    return F


def make_w(spec: TBATSSpec, phi: jnp.ndarray) -> jnp.ndarray:
    d = spec.state_dim
    w = jnp.zeros(d)
    w = w.at[0].set(1.0)
    if spec.use_trend:
        w = w.at[1].set(phi)
        start_seasonal = 2
    else:
        start_seasonal = 1

    for i in range(spec.n_gamma):
        w = w.at[start_seasonal + 2 * i].set(1.0)  # cos position
    return w


def make_g(spec: TBATSSpec, params: Params) -> jnp.ndarray:
    d = spec.state_dim
    g = jnp.zeros(d)
    g = g.at[0].set(params.alpha)
    if spec.use_trend:
        g = g.at[1].set(params.beta)
        start_seasonal = 2
    else:
        start_seasonal = 1

    for i in range(spec.n_gamma):
        g = g.at[start_seasonal + 2 * i].set(params.gamma1[i])
        g = g.at[start_seasonal + 2 * i + 1].set(params.gamma2[i])
    return g


def build_matrices(spec: TBATSSpec, params: Params):
    F = make_F(spec, params.phi)
    w = make_w(spec, params.phi)
    g = make_g(spec, params)
    return F, g, w
