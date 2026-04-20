"""Point forecast: propagate state deterministically (no innovations).

If the spec enables Box-Cox, the forecast is produced on the transformed
scale and inverse-transformed before return.
"""

import jax.numpy as jnp
from jax import lax

from tbats_jax.spec import TBATSSpec
from tbats_jax.params import unpack
from tbats_jax.matrices import build_matrices
from tbats_jax.kernel import tbats_scan, tbats_scan_masked
from tbats_jax.boxcox import boxcox, inv_boxcox


def forecast(y, theta, spec: TBATSSpec, horizon: int):
    """Return array of length `horizon` with point forecasts on the original scale."""
    params = unpack(theta, spec)
    F, g, w = build_matrices(spec, params)

    import numpy as _np
    y = jnp.asarray(y)
    has_missing = bool(_np.isnan(_np.asarray(y)).any())
    if spec.use_box_cox:
        if has_missing:
            finite = ~jnp.isnan(y)
            safe = jnp.where(finite, y, 1.0)
            y_eff = jnp.where(finite, boxcox(safe, params.box_cox_lambda), jnp.nan)
        else:
            y_eff = boxcox(y, params.box_cox_lambda)
    else:
        y_eff = y
    scan_fn = tbats_scan_masked if has_missing else tbats_scan
    _, x_T = scan_fn(y_eff, params.x0, F, g, w)

    def step(x_prev, _):
        y_hat = jnp.dot(w, x_prev)
        x_next = F @ x_prev
        return x_next, y_hat

    _, preds = lax.scan(step, x_T, jnp.zeros(horizon))
    if spec.use_box_cox:
        preds = inv_boxcox(preds, params.box_cox_lambda)
    return preds
