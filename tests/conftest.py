"""Test setup: make the repo importable and keep the data root out of the way.

The tests here cover pure functions only — no test reads the data tree. But
`collective_functions` computes `paths.DATA_ENVIRONMENT` at import time, so
`RL_DATA_ROOT` is pointed at a temp dir to be sure a test can never touch the
real data.

Note on running: a system ROS install may ship pytest plugins that fail to
import (launch_testing -> yaml). If collection dies before reaching this file,
run with autoloading off:

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest
"""

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("RL_DATA_ROOT", tempfile.mkdtemp(prefix="rl-tests-"))

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
