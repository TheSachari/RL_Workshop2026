"""Run a repo script with deterministic RNG, from a contained workspace.

    python tools/golden/run_golden.py --workspace <dir> -- simulation_start.py --dataset ...

The target script is executed with `run_name="__main__"` so its `if __name__ ==
"__main__"` block fires, exactly as a normal CLI invocation would.

Containment
-----------
The scripts navigate with `os.chdir` and write results to `../Plots` *relative to
the data directory they last changed into*. Run from the wrong place, a run
silently writes outside its working directory. This runner cds into `--workspace`
first, so a workspace whose `Data_environment/` is a real directory (of symlinks
to the read-only inputs) keeps every artifact inside the workspace.
"""

import argparse
import os
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import determinism  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, help="dir to run from")
    parser.add_argument("script", help="repo script, e.g. simulation_start.py")
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    script = REPO_ROOT / args.script
    if not script.is_file():
        parser.error(f"script not found: {script}")

    workspace = Path(args.workspace).resolve()
    if not workspace.is_dir():
        parser.error(f"workspace not found: {workspace}")

    seed = determinism.apply()
    print(f"[golden] seed={seed} workspace={workspace}", flush=True)

    sys.path.insert(0, str(REPO_ROOT))
    os.chdir(workspace)

    script_args = [a for a in args.script_args if a != "--"]
    sys.argv = [str(script), *script_args]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
