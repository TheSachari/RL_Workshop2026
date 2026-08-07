"""Build a contained run workspace that symlinks a read-only data directory.

    python tools/golden/make_workspace.py --data-root <dir> --workspace <dir>

`--data-root` is a directory holding the project's `Data*/`, `Reward_weights/`
and `SVG_model/` folders (i.e. a tree already produced by `preprocess.py`).
Nothing there is written to or modified.

`Data_environment` is created as a *real* directory of per-file symlinks rather
than a symlink to the source folder. The scripts save results to `../Plots`
relative to it; if it were a symlink, `..` would resolve into the source tree and
runs would write their outputs there.
"""

import argparse
from pathlib import Path

LINKED_DIRS = [
    "Data",
    "Data_preprocessed",
    "Data_sampled",
    "Data_trained",
    "Reward_weights",
    "SVG_model",
]
# Real dir of file symlinks, so `../Plots` stays inside the workspace.
FILE_LINKED_DIRS = ["Data_environment"]
OUTPUT_DIRS = ["Plots"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--workspace", required=True)
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    workspace = Path(args.workspace).resolve()
    if not data_root.is_dir():
        parser.error(f"data root not found: {data_root}")

    workspace.mkdir(parents=True, exist_ok=True)

    for name in LINKED_DIRS:
        src, dst = data_root / name, workspace / name
        if dst.is_symlink() or dst.exists():
            dst.unlink() if dst.is_symlink() else None
        if src.is_dir():
            dst.symlink_to(src)
        else:
            print(f"[skip] missing in data root: {name}")

    for name in FILE_LINKED_DIRS:
        src, dst = data_root / name, workspace / name
        if dst.is_symlink():
            dst.unlink()
        dst.mkdir(parents=True, exist_ok=True)
        if not src.is_dir():
            print(f"[skip] missing in data root: {name}")
            continue
        for item in src.iterdir():
            link = dst / item.name
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(item)

    for name in OUTPUT_DIRS:
        (workspace / name).mkdir(parents=True, exist_ok=True)

    print(f"[ok] workspace ready: {workspace}")


if __name__ == "__main__":
    main()
