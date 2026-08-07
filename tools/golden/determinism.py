"""Seed the process RNGs before a golden run.

Imported by `run_golden.py` before the target script is executed.

This used to also monkeypatch `DataFrame.sample` to inject a `random_state`,
because `load_environment_variables` sampled the skill table without one and
runs were therefore not reproducible. That is now fixed in the loader itself
(it takes an explicit `seed`), so only ordinary RNG seeding remains here — and
the scripts seed themselves from `--seed` anyway.

GPU note
--------
The agent case runs on CUDA (5x faster than CPU here, and reproducible run to
run on this machine). Floating-point accumulation order differs between
backends, so **a reference recorded on GPU does not match a CPU run** — around
half the metrics move. Re-record with `run_cases.py record` after changing
device or GPU. `CUBLAS_WORKSPACE_CONFIG` must be set before the first CUDA
context is created, hence at import time rather than inside `apply()`.
"""

import os
import random

import numpy as np

SEED = int(os.environ.get("GOLDEN_SEED", "42"))

# Required for deterministic cuBLAS matmuls; must precede CUDA initialisation.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def apply() -> int:
    """Seed the global RNGs. Returns the seed."""
    random.seed(SEED)
    np.random.seed(SEED)

    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    return SEED
