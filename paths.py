"""Absolute paths to the project's data directories.

Replaces the `os.chdir` navigation the scripts used to rely on. Import the
directory you need and join onto it:

    from paths import DATA_ENVIRONMENT
    df = pd.read_pickle(DATA_ENVIRONMENT / "df_stations.pkl")

Why not `os.chdir`
------------------
Changing the process-wide working directory to reach a file makes every later
relative path depend on execution order. Concretely, in this project it meant
modules could not be imported without the folders already existing, results were
written relative to whichever directory was last entered (so a run launched from
the wrong place wrote into the source data tree), and two runs in the same
process or in parallel interfered with each other.

Where the data lives
--------------------
By default, next to the repo (`<repo>/Data`, `<repo>/Data_environment`, ...).
Point `RL_DATA_ROOT` elsewhere to use a data tree stored outside the repo:

    export RL_DATA_ROOT=/path/to/data
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Data lives outside the repo (it is large and not versioned), so allow an
# override. Defaults to the repo root, which is where preprocess.py creates it.
DATA_ROOT = Path(os.environ.get("RL_DATA_ROOT", REPO_ROOT)).resolve()

DATA = DATA_ROOT / "Data"
DATA_PREPROCESSED = DATA_ROOT / "Data_preprocessed"
DATA_TRAINED = DATA_ROOT / "Data_trained"
DATA_SAMPLED = DATA_ROOT / "Data_sampled"
DATA_ENVIRONMENT = DATA_ROOT / "Data_environment"
SVG_MODEL = DATA_ROOT / "SVG_model"
PLOTS = DATA_ROOT / "Plots"
REWARD_WEIGHTS = DATA_ROOT / "Reward_weights"

# Created by preprocess.py on first run; listed here so every consumer agrees.
ALL_DIRS = [
    DATA_PREPROCESSED,
    DATA_TRAINED,
    DATA_SAMPLED,
    DATA_ENVIRONMENT,
    SVG_MODEL,
    PLOTS,
    REWARD_WEIGHTS,
]


def ensure_dirs() -> None:
    """Create the output directories if they do not exist."""
    for directory in ALL_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


def resolve(path_like, default_dir: Path) -> Path:
    """Resolve a user-supplied path, falling back to `default_dir` for bare names.

    Lets CLI arguments stay as plain filenames (`--dataset df_pc_real_prob.pkl`)
    while still accepting an absolute or explicitly relative path.
    """
    path = Path(path_like)
    if path.is_absolute() or len(path.parts) > 1:
        return path.resolve()
    return default_dir / path
