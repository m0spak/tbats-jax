"""Build TBATS state-space matrices F, g, w from (spec, params).

State layout:
  [level, (slope), seasonal (2*n_gamma interleaved cos/sin),
   AR lags (p), MA lags (q)]

Transition F (for the full ARMA case):

  level row   [0]:     [1, phi,  0*_seasonal,         alpha*ar_1..p,  alpha*ma_1..q]
  slope row   [1]:     [0, phi,  0*_seasonal,         beta*ar_1..p,   beta*ma_1..q]
  seasonal blocks:     2x2 rotation per harmonic (interleaved).
                       Coupling with AR/MA: seasonal row gets gamma_j*ar_j etc.
  AR block (p rows):   [0,_, 0*_seasonal,  companion(ar),  [ma_1..q ; 0]_{p x q}]
  MA block (q rows):   [0,_, 0*_seasonal,  0,              companion(0; shift-identity)]

Observation w (y_hat = w @ x_{t-1}):
  [1, phi, 1, 0, 1, 0, ..., 1, 0,  ar_1..p,  ma_1..q]
  — level + phi (if damp) + cos-only seasonal + AR coefs + MA coefs.

Innovation g (x_t = F @ x_{t-1} + g * e_t):
  [alpha, beta, gamma1_j, gamma2_j, ...,  1 (at first AR), 0 (rest AR),
                                           1 (at first MA), 0 (rest MA)]
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


def _ar_start(spec: TBATSSpec) -> int:
    return 1 + int(spec.use_trend) + 2 * spec.n_gamma


def _ma_start(spec: TBATSSpec) -> int:
    return _ar_start(spec) + spec.p


def _seasonal_start(spec: TBATSSpec) -> int:
    return 1 + int(spec.use_trend)


def make_F(spec: TBATSSpec, params: Params) -> jnp.ndarray:
    d = spec.state_dim
    phi = params.phi
    alpha = params.alpha
    beta = params.beta
    ar = params.ar  # (p,)
    ma = params.ma  # (q,)
    F = jnp.zeros((d, d))

    # Level and trend rows
    F = F.at[0, 0].set(1.0)
    if spec.use_trend:
        F = F.at[0, 1].set(phi)
        F = F.at[1, 1].set(phi)

    # Seasonal rotation blocks
    s_start = _seasonal_start(spec)
    lambdas = _seasonal_lambdas(spec)
    for i, lam in enumerate(lambdas):
        pos = s_start + 2 * i
        c = jnp.cos(lam)
        s = jnp.sin(lam)
        F = F.at[pos, pos].set(c)
        F = F.at[pos, pos + 1].set(s)
        F = F.at[pos + 1, pos].set(-s)
        F = F.at[pos + 1, pos + 1].set(c)

    # AR columns contribute to level (alpha*ar) and trend (beta*ar)
    ar_start = _ar_start(spec)
    for j in range(spec.p):
        F = F.at[0, ar_start + j].set(alpha * ar[j])
        if spec.use_trend:
            F = F.at[1, ar_start + j].set(beta * ar[j])

    # MA columns similarly
    ma_start = _ma_start(spec)
    for j in range(spec.q):
        F = F.at[0, ma_start + j].set(alpha * ma[j])
        if spec.use_trend:
            F = F.at[1, ma_start + j].set(beta * ma[j])

    # Seasonal coupling with AR/MA (matching R's makeMatrices.R lines 105-112)
    #   seasonal row gets  gamma_{cos/sin at that harmonic} * ar_j / ma_j
    if spec.n_gamma > 0 and (spec.p > 0 or spec.q > 0):
        # For each seasonal row, the innovation contribution via AR/MA column
        # is gamma_entry * ar_coef (or ma_coef). "gamma_entry" follows the
        # g-vector's seasonal layout: interleaved gamma1/gamma2 per harmonic.
        for i in range(spec.n_gamma):
            pos_cos = s_start + 2 * i
            pos_sin = s_start + 2 * i + 1
            g1 = params.gamma1[i]
            g2 = params.gamma2[i]
            for j in range(spec.p):
                F = F.at[pos_cos, ar_start + j].set(g1 * ar[j])
                F = F.at[pos_sin, ar_start + j].set(g2 * ar[j])
            for j in range(spec.q):
                F = F.at[pos_cos, ma_start + j].set(g1 * ma[j])
                F = F.at[pos_sin, ma_start + j].set(g2 * ma[j])

    # AR companion block: first row = ar_coefs; rows 1..p-1 = shift-identity
    if spec.p > 0:
        # First AR row gets ar coefs across AR cols
        for j in range(spec.p):
            F = F.at[ar_start, ar_start + j].set(ar[j])
        # Rows 1..p-1: shift (x_t[ar+i] = x_{t-1}[ar+i-1])
        for i in range(1, spec.p):
            F = F.at[ar_start + i, ar_start + i - 1].set(1.0)
        # First AR row also couples to MA cols via ma_coefs
        for j in range(spec.q):
            F = F.at[ar_start, ma_start + j].set(ma[j])

    # MA companion block: shift only (rows 1..q-1 shift-identity)
    if spec.q > 0:
        for i in range(1, spec.q):
            F = F.at[ma_start + i, ma_start + i - 1].set(1.0)

    return F


def make_w(spec: TBATSSpec, params: Params) -> jnp.ndarray:
    d = spec.state_dim
    w = jnp.zeros(d)
    w = w.at[0].set(1.0)
    if spec.use_trend:
        w = w.at[1].set(params.phi)

    s_start = _seasonal_start(spec)
    for i in range(spec.n_gamma):
        w = w.at[s_start + 2 * i].set(1.0)  # cos positions

    ar_start = _ar_start(spec)
    for j in range(spec.p):
        w = w.at[ar_start + j].set(params.ar[j])

    ma_start = _ma_start(spec)
    for j in range(spec.q):
        w = w.at[ma_start + j].set(params.ma[j])

    return w


def make_g(spec: TBATSSpec, params: Params) -> jnp.ndarray:
    d = spec.state_dim
    g = jnp.zeros(d)
    g = g.at[0].set(params.alpha)
    if spec.use_trend:
        g = g.at[1].set(params.beta)

    s_start = _seasonal_start(spec)
    for i in range(spec.n_gamma):
        g = g.at[s_start + 2 * i].set(params.gamma1[i])
        g = g.at[s_start + 2 * i + 1].set(params.gamma2[i])

    # Innovation enters AR[0] and MA[0] as a 1; shift does the rest.
    if spec.p > 0:
        g = g.at[_ar_start(spec)].set(1.0)
    if spec.q > 0:
        g = g.at[_ma_start(spec)].set(1.0)

    return g


def build_matrices(spec: TBATSSpec, params: Params):
    F = make_F(spec, params)
    w = make_w(spec, params)
    g = make_g(spec, params)
    return F, g, w
