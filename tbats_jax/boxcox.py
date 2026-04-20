"""Box-Cox transform (positive data only).

For lambda != 0: (y^lambda - 1) / lambda
For lambda == 0: log(y)

We use the stable form `expm1(lam * log(y)) / lam`, which is well-defined
for lam > 0 (our bounded range) and numerically stable as lam -> 0.

Reference: forecast::BoxCox in R/forecast2.R.
"""

import jax.numpy as jnp


def boxcox(y, lam):
    """Forward transform. y must be > 0."""
    ly = jnp.log(y)
    # expm1(lam * ly) / lam == (y^lam - 1) / lam, stable for small lam
    return jnp.expm1(lam * ly) / lam


def inv_boxcox(z, lam):
    """Inverse transform of boxcox output."""
    # (1 + lam*z)^(1/lam)  ==  exp(log1p(lam*z) / lam)
    return jnp.exp(jnp.log1p(lam * z) / lam)


def boxcox_log_jacobian(y, lam):
    """log |d/dy BoxCox(y, lam)| summed over observed points (NaNs skipped).

    d/dy (y^lam - 1)/lam = y^(lam - 1)
    log |.| = (lam - 1) * log(y)
    """
    log_y = jnp.log(jnp.where(jnp.isnan(y), 1.0, y))
    mask = (~jnp.isnan(y)).astype(log_y.dtype)
    return (lam - 1.0) * jnp.sum(log_y * mask)
