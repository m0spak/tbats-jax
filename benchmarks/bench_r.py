"""Python shim around bench_r.R — subprocesses Rscript, parses JSON."""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


HERE = Path(__file__).parent
R_SCRIPT = HERE / "bench_r.R"


def r_available() -> bool:
    return shutil.which("Rscript") is not None


def run_r_bench(
    y: np.ndarray,
    mode: str,
    use_trend: bool,
    use_damping: bool,
    periods,
    k_vector,
    h: int = 0,
    use_box_cox: bool = False,
    timeout_s: int = 600,
) -> dict:
    """Write series to tmp, shell Rscript, parse JSON back.

    Returns the dict from bench_r.R plus stdout/stderr for debugging.
    """
    if not r_available():
        raise RuntimeError("Rscript not on PATH — brew install r")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        series_path = td / "series.csv"
        out_path = td / "out.json"
        np.savetxt(series_path, np.asarray(y, dtype=np.float64), fmt="%.15g")

        cmd = [
            "Rscript", str(R_SCRIPT),
            str(series_path), str(out_path),
            mode,
            "TRUE" if use_trend else "FALSE",
            "TRUE" if use_damping else "FALSE",
            ",".join(str(p) for p in periods),
            ",".join(str(k) for k in k_vector),
            str(int(h)),
            "TRUE" if use_box_cox else "FALSE",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        if res.returncode != 0 or not out_path.exists():
            raise RuntimeError(
                f"Rscript failed (exit {res.returncode})\nstderr:\n{res.stderr}"
            )
        data = json.loads(out_path.read_text())

    return data
