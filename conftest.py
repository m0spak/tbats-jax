"""Enable float64 globally and silence GPU/Metal-absence warnings."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
