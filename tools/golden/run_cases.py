"""Record or verify the golden reference runs.

    # once, before touching any code:
    python tools/golden/run_cases.py record --data-root <dir>

    # after every refactor step:
    python tools/golden/run_cases.py check --data-root <dir>

`record` runs each case twice and refuses to store a reference unless both runs
agree — a reference that is not itself reproducible is worthless.

`check` re-runs each case and compares against the stored reference. Non-zero
exit means the refactor changed behaviour.
"""

import argparse
import json
import pickle
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
GOLDEN_DIR = HERE / "reference"
CASES = json.loads((HERE / "cases.json").read_text())["cases"]


def build_workspace(data_root: Path, workspace: Path) -> None:
    subprocess.run(
        [sys.executable, str(HERE / "make_workspace.py"),
         "--data-root", str(data_root), "--workspace", str(workspace)],
        check=True, capture_output=True,
    )


def run_case(case: dict, workspace: Path, tag: str) -> dict:
    """Execute one case; return its metrics dict."""
    cmd = [
        sys.executable, str(HERE / "run_golden.py"),
        "--workspace", str(workspace),
        case["script"], *case["args"],
        "--save_metrics_as", tag,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"case {case['name']} failed (exit {proc.returncode}):\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )
    out = workspace / "Plots" / f"{tag}.pkl"
    if not out.is_file():
        raise RuntimeError(f"case {case['name']} produced no metrics at {out}")
    with out.open("rb") as fh:
        return pickle.load(fh)


def diff(a: dict, b: dict) -> list[str]:
    keys = sorted(set(a) | set(b))
    return [f"{k}: {a.get(k, '<missing>')} != {b.get(k, '<missing>')}"
            for k in keys if a.get(k) != b.get(k)]


def cmd_record(args) -> int:
    workspace = args.workspace.resolve()
    build_workspace(args.data_root.resolve(), workspace)
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    failures = 0
    for case in CASES:
        name = case["name"]
        print(f"[record] {name} ... ", end="", flush=True)
        first = run_case(case, workspace, f"_rec1_{name}")
        second = run_case(case, workspace, f"_rec2_{name}")

        drift = diff(first, second)
        if drift:
            failures += 1
            print("NOT REPRODUCIBLE - not recorded")
            for line in drift:
                print(f"    {line}")
            continue

        payload = {
            "case": name,
            "script": case["script"],
            "args": case["args"],
            "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "metrics": first,
        }
        (GOLDEN_DIR / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(f"ok ({len(first)} metrics)")

    print()
    print(f"recorded {len(CASES) - failures}/{len(CASES)} cases into {GOLDEN_DIR}")
    return 1 if failures else 0


def cmd_check(args) -> int:
    workspace = args.workspace.resolve()
    build_workspace(args.data_root.resolve(), workspace)

    failures = 0
    for case in CASES:
        name = case["name"]
        ref_path = GOLDEN_DIR / f"{name}.json"
        if not ref_path.is_file():
            print(f"[check] {name} ... NO REFERENCE (run `record` first)")
            failures += 1
            continue

        print(f"[check] {name} ... ", end="", flush=True)
        got = run_case(case, workspace, f"_chk_{name}")
        ref = json.loads(ref_path.read_text())["metrics"]
        drift = diff(ref, got)
        if drift:
            failures += 1
            print("REGRESSION")
            for line in drift:
                print(f"    {line}")
        else:
            print("ok")

    print()
    if failures:
        print(f"FAILED: {failures}/{len(CASES)} case(s) changed behaviour")
        return 1
    print(f"PASS: {len(CASES)}/{len(CASES)} cases match the golden reference")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["record", "check"])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--workspace", type=Path,
        default=REPO_ROOT.parent / "golden" / "workspace",
    )
    args = parser.parse_args()
    return cmd_record(args) if args.mode == "record" else cmd_check(args)


if __name__ == "__main__":
    sys.exit(main())
