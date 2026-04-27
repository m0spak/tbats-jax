"""JAX-based TBATS: fixed-structure innovations state-space fit + forecast."""

from tbats_jax.spec import TBATSSpec
from tbats_jax.fit import fit
from tbats_jax.fit_jax import fit_jax, fit_panel, fit_panel_hetero
from tbats_jax.fit_scan import fit_scan, fit_panel_scan
from tbats_jax.fit_lm import fit_lm, fit_lm_multistart, fit_panel_lm
from tbats_jax.auto import auto_fit_jax, auto_fit_jax_cv, AutoResult, AutoCVResult
# Bayesian path is optional (requires numpyro). Load lazily so a plain
# `import tbats_jax` doesn't fail when numpyro isn't installed.


_BAYES_NAMES = ("bayes_tbats", "svi_tbats", "bayes_forecast", "BayesResult")


def __getattr__(name):
    if name in _BAYES_NAMES:
        import importlib
        mod = importlib.import_module("tbats_jax.bayesian")
        return getattr(mod, name)
    raise AttributeError(f"module 'tbats_jax' has no attribute {name!r}")
from tbats_jax.kernel import tbats_scan, neg_log_likelihood
from tbats_jax.forecast import forecast

__all__ = [
    "TBATSSpec",
    "fit",
    "fit_jax",
    "fit_panel",
    "fit_panel_hetero",
    "fit_scan",
    "fit_panel_scan",
    "fit_lm",
    "fit_lm_multistart",
    "fit_panel_lm",
    "auto_fit_jax",
    "auto_fit_jax_cv",
    "AutoResult",
    "AutoCVResult",
    "bayes_tbats",
    "svi_tbats",
    "bayes_forecast",
    "BayesResult",
    "forecast",
    "tbats_scan",
    "neg_log_likelihood",
]
