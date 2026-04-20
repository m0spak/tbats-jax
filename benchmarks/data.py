"""Thin re-export shim. Data generators live in `tbats_jax.datasets` now
(so they're importable after `pip install tbats-jax`). This module is kept
so existing bench scripts and notebooks that imported from
`benchmarks.data` keep working unchanged.
"""

from tbats_jax.datasets import (
    synthesize_daily,
    synthesize_two_seasonal,
)

__all__ = ["synthesize_daily", "synthesize_two_seasonal"]
