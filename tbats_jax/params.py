"""Parameter vector packing / unpacking.

Layout (matching forecast::tbats fitTBATS.R ordering):
  [(lambda), alpha, (phi), (beta), gamma1[1..K], gamma2[1..K], x0[1..state_dim]]
where K = n_gamma = sum of harmonic counts. Optional blocks appear only if
the spec enables them.
"""

from typing import NamedTuple, Tuple
import jax.numpy as jnp
import numpy as np

from tbats_jax.spec import TBATSSpec


class Params(NamedTuple):
    box_cox_lambda: jnp.ndarray  # scalar; 1.0 (no-op) when Box-Cox disabled
    alpha: jnp.ndarray
    phi: jnp.ndarray   # scalar; 1.0 when damping disabled
    beta: jnp.ndarray  # scalar; 0.0 when trend disabled
    gamma1: jnp.ndarray  # shape (n_gamma,)
    gamma2: jnp.ndarray  # shape (n_gamma,)
    x0: jnp.ndarray      # shape (state_dim,)


def unpack(theta: jnp.ndarray, spec: TBATSSpec) -> Params:
    i = 0
    if spec.use_box_cox:
        box_cox_lambda = theta[i]; i += 1
    else:
        box_cox_lambda = jnp.array(1.0)
    alpha = theta[i]; i += 1
    if spec.use_damping:
        phi = theta[i]; i += 1
    else:
        phi = jnp.array(1.0)
    if spec.use_trend:
        beta = theta[i]; i += 1
    else:
        beta = jnp.array(0.0)
    K = spec.n_gamma
    gamma1 = theta[i:i + K]; i += K
    gamma2 = theta[i:i + K]; i += K
    x0 = theta[i:i + spec.state_dim]
    return Params(box_cox_lambda, alpha, phi, beta, gamma1, gamma2, x0)


def init_theta(spec: TBATSSpec, y: np.ndarray) -> np.ndarray:
    """Starting values matching forecast::fitTBATS (lines 160-191):
    lambda=0.5 if Box-Cox, alpha=0.09, phi=0.999, beta=0.05, gammas=0.
    """
    theta = []
    if spec.use_box_cox:
        theta.append(0.5)
    theta.append(0.09)
    if spec.use_damping:
        theta.append(0.999)
    if spec.use_trend:
        theta.append(0.05)
    theta.extend([0.0] * spec.n_gamma)
    theta.extend([0.0] * spec.n_gamma)
    x0 = np.zeros(spec.state_dim)
    # Warmup: first 10 finite observations (NaN-robust version of old init).
    y_arr = np.asarray(y, dtype=np.float64)
    finite = y_arr[np.isfinite(y_arr)]
    window = finite[: min(len(finite), 10)] if finite.size else np.array([0.0])
    if spec.use_box_cox:
        lam = 0.5
        pos = np.where(window > 0, window, 1e-6)
        window_eff = (pos ** lam - 1.0) / lam
    else:
        window_eff = window
    x0[0] = float(np.mean(window_eff))
    theta.extend(x0.tolist())
    return np.asarray(theta, dtype=np.float64)


def param_names(spec: TBATSSpec) -> Tuple[str, ...]:
    names = ["alpha"]
    if spec.use_damping:
        names.append("phi")
    if spec.use_trend:
        names.append("beta")
    for idx, (m, k) in enumerate(spec.seasonal):
        for j in range(1, k + 1):
            names.append(f"gamma1_s{idx}_j{j}")
    for idx, (m, k) in enumerate(spec.seasonal):
        for j in range(1, k + 1):
            names.append(f"gamma2_s{idx}_j{j}")
    names.append("x0_level")
    if spec.use_trend:
        names.append("x0_slope")
    for idx, (m, k) in enumerate(spec.seasonal):
        for j in range(1, k + 1):
            names.append(f"x0_s{idx}_j{j}_cos")
            names.append(f"x0_s{idx}_j{j}_sin")
    return tuple(names)
