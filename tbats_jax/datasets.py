"""Public datasets module — synthetic generators + optional real-data fetcher.

Two tiers:

  **Synthetic**  (`synthesize_daily`, `synthesize_two_seasonal`) — deterministic
  reproducible series generated at runtime from a seed. No data is bundled in
  the wheel; each call builds the series fresh.

  **Real data fetcher**  (`fetch_taylor`) — downloads the canonical half-hourly
  UK electricity demand series *from its original GPL-3 source* to the user's
  local cache. `tbats-jax` itself is MIT-licensed and **never redistributes**
  the Taylor data. The fetcher respects upstream licensing by leaving the
  downloaded bytes on the user's machine — same model as HuggingFace Hub or
  torchvision's dataset downloaders.

  Requires: `pyreadr` (install via `pip install tbats-jax[data]`).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Synthetic generators — no external deps, no bundled data.
# ---------------------------------------------------------------------------

def synthesize_two_seasonal(
    n: int = 2000,
    periods=(24.0, 168.0),
    amps=((3.0, 1.5), (2.0, 0.8)),
    trend_slope: float = 0.002,
    noise_sd: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    """Hourly-style series: linear trend + two seasonal sinusoids + white noise.

    Matches the shape used by our GPU panel benchmarks. Defaults produce a
    ~2000-point series with periods 24 and 168 (hourly + weekly).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    y = 10.0 + trend_slope * t
    for (m, (a1, a2)) in zip(periods, amps):
        y = y + a1 * np.sin(2 * np.pi * t / m) + a2 * np.cos(4 * np.pi * t / m)
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


# ---------------------------------------------------------------------------
# Real-data fetcher — downloads GPL-3 source on demand, NEVER redistributed.
# ---------------------------------------------------------------------------

_TAYLOR_URL = "https://github.com/robjhyndman/forecast/raw/main/data/taylor.rda"


def fetch_taylor(cache_dir: str = None, url: str = _TAYLOR_URL) -> np.ndarray:
    """Download Taylor (UK half-hourly electricity demand, 2000-06-05..2000-08-27).

    Returns the 4032-point series as a float64 numpy array.

    Licensing notes
    ---------------
    The Taylor dataset is distributed by R's `forecast` package under the
    **GPL-3** license. `tbats-jax` is MIT-licensed and does NOT include this
    data in its wheel. This function downloads the data **from the upstream
    GPL-3 source** to your local machine (default cache: ``~/.cache/tbats_jax/``).
    The cached bytes on your disk remain under their original GPL-3 terms;
    `tbats-jax`'s MIT license applies only to this fetcher's source code.

    Parameters
    ----------
    cache_dir : str, optional
        Where to cache the downloaded ``taylor.rda`` (and a parsed CSV for
        fast reloading). Defaults to ``~/.cache/tbats_jax/``.
    url : str, optional
        Override the upstream URL (e.g. for a private mirror or offline
        copy).

    Requires
    --------
    The optional `pyreadr` dependency — install with
    ``pip install "tbats-jax[data]"``.
    """
    import os
    import urllib.request

    try:
        import pyreadr  # type: ignore
    except ImportError as e:
        raise ImportError(
            "fetch_taylor requires the optional `pyreadr` dependency. "
            'Install with: pip install "tbats-jax[data]"'
        ) from e

    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/tbats_jax")
    os.makedirs(cache_dir, exist_ok=True)
    rda_path = os.path.join(cache_dir, "taylor.rda")
    csv_path = os.path.join(cache_dir, "taylor.csv")

    # Fast path: parsed CSV already cached.
    if os.path.exists(csv_path):
        return np.loadtxt(csv_path, dtype=np.float64)

    # Slow path: download .rda (first time only), parse, cache as CSV.
    if not os.path.exists(rda_path):
        urllib.request.urlretrieve(url, rda_path)

    obj = pyreadr.read_r(rda_path)
    # forecast::taylor is a 1-column 'ts' object — pyreadr returns a DataFrame.
    df = obj["taylor"]
    y = np.asarray(df.iloc[:, 0], dtype=np.float64)

    np.savetxt(csv_path, y, fmt="%.12g")
    return y


__all__ = [
    "synthesize_two_seasonal",
    "synthesize_daily",
    "fetch_taylor",
]
