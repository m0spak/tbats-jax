"""Tests for the public `tbats_jax.datasets` module."""

import numpy as np
import pytest

from tbats_jax.datasets import (
    synthesize_daily,
    synthesize_two_seasonal,
    fetch_taylor,
)


def test_synthesize_daily_shape_and_reproducibility():
    y1 = synthesize_daily(n=365, seed=0)
    y2 = synthesize_daily(n=365, seed=0)
    assert y1.shape == (365,)
    assert np.array_equal(y1, y2), "same seed must produce identical output"
    y3 = synthesize_daily(n=365, seed=1)
    assert not np.array_equal(y1, y3), "different seeds must differ"


def test_synthesize_two_seasonal_shape_and_reproducibility():
    y1 = synthesize_two_seasonal(n=500, seed=0)
    y2 = synthesize_two_seasonal(n=500, seed=0)
    assert y1.shape == (500,)
    assert np.array_equal(y1, y2)


def test_synthesize_finite_and_varied():
    """No NaN / inf, and variance is non-trivial (the generator isn't broken)."""
    y = synthesize_daily(n=500, seed=42)
    assert np.all(np.isfinite(y))
    assert y.std() > 1.0

    y2 = synthesize_two_seasonal(n=500, seed=42)
    assert np.all(np.isfinite(y2))
    assert y2.std() > 0.5


def test_fetch_taylor_signals_missing_pyreadr():
    """If pyreadr isn't installed, fetch_taylor should raise ImportError
    with a clear install hint — not a cryptic attribute/module error.

    We skip this test when pyreadr IS installed (locally the project env
    doesn't have it; on CI we'd match that)."""
    try:
        import pyreadr  # noqa: F401
        pytest.skip("pyreadr installed — error-path test not applicable")
    except ImportError:
        pass

    with pytest.raises(ImportError, match=r"pyreadr"):
        fetch_taylor()
