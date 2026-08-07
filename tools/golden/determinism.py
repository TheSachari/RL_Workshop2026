"""Force every RNG the simulation touches into a known state.

Imported by `run_golden.py` before the target script is executed, so the repo's
own sources stay untouched at this stage of the refactor.

Why this is needed
------------------
`collective_functions.load_environment_variables` calls

    df_skills = df_skills.sample(len(df_skills) // constraint_factor_ff)

with no `random_state`. `DataFrame.sample` draws from the *numpy* global RNG,
which `random.seed(42)` (called in `constrain_veh`) does not control. Two runs of
the same command therefore use a different set of firefighters and produce
different metrics.

Note that even `--constraint_factor_ff 1` is affected: the selected *set* is the
whole table, but `sample` returns it in a shuffled *order*, and role assignment
scans firefighters in order.

Once the refactor threads an explicit seed through the loader, this shim can be
deleted.
"""

import os
import random

import numpy as np
import pandas as pd

SEED = int(os.environ.get("GOLDEN_SEED", "42"))

_orig_df_sample = pd.DataFrame.sample
_orig_series_sample = pd.Series.sample


def _df_sample(self, *args, **kwargs):
    kwargs.setdefault("random_state", SEED)
    return _orig_df_sample(self, *args, **kwargs)


def _series_sample(self, *args, **kwargs):
    kwargs.setdefault("random_state", SEED)
    return _orig_series_sample(self, *args, **kwargs)


def apply() -> int:
    """Seed all RNGs and make pandas' `.sample` deterministic. Returns the seed."""
    random.seed(SEED)
    np.random.seed(SEED)

    pd.DataFrame.sample = _df_sample
    pd.Series.sample = _series_sample

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
