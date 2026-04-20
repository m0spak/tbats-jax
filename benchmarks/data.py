"""Synthetic series for benchmarking. Reproducible, multi-seasonal."""

import numpy as np


def synthesize_two_seasonal(
    n: int = 2000,
    periods=(24.0, 168.0),
    amps=((3.0, 1.5), (2.0, 0.8)),
    trend_slope: float = 0.002,
    noise_sd: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    """Level + linear trend + two seasonal sinusoids + white noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    y = 10.0 + trend_slope * t
    for (m, (a1, a2)) in zip(periods, amps):
        y = y + a1 * np.sin(2 * np.pi * t / m) + a2 * np.cos(2 * np.pi * t / m * 2)
    y = y + rng.normal(0.0, noise_sd, size=n)
    return y


def synthesize_daily(n: int = 730, seed: int = 0) -> np.ndarray:
    """Daily series with weekly + yearly seasonality — the typical retail /
    utility shape. 2 years by default. Trend + weekly + yearly + noise.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    y = 100.0 + 0.02 * t
    y += 8.0 * np.sin(2 * np.pi * t / 7.0) + 3.0 * np.cos(4 * np.pi * t / 7.0)
    y += 15.0 * np.sin(2 * np.pi * t / 365.25) + 5.0 * np.cos(4 * np.pi * t / 365.25)
    y += rng.normal(0, 2.0, n)
    return y
