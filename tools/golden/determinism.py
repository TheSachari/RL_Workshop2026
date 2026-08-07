"""Seed the process RNGs before a golden run.

Imported by `run_golden.py` before the target script is executed.

This used to also monkeypatch `DataFrame.sample` to inject a `random_state`,
because `load_environment_variables` sampled the skill table without one and
runs were therefore not reproducible. That is now fixed in the loader itself
(it takes an explicit `seed`), so only ordinary RNG seeding remains here — and
the scripts seed themselves from `--seed` anyway.
"""

import os
import random

import numpy as np

SEED = int(os.environ.get("GOLDEN_SEED", "42"))


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

    return SEED
