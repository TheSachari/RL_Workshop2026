"""Compare a metrics pickle against a recorded golden reference.

    python tools/golden/compare.py golden/<case>.json Plots/<case>.pkl

Exit code 0 when every metric matches, 1 otherwise. Use it after each refactor
step: the numbers must not move.

Metrics are plain scalars, so the comparison is exact by default. `--tol` allows
a relative tolerance for the float ones (`skill_lvl`) if a step legitimately
reorders floating-point accumulation.
"""

import argparse
import json
import math
import pickle
import sys
from pathlib import Path


def load_metrics(path: Path) -> dict:
    if path.suffix == ".json":
        return json.loads(path.read_text())["metrics"]
    with path.open("rb") as fh:
        return pickle.load(fh)


def compare(ref: dict, got: dict, tol: float) -> list[str]:
    problems = []

    for key in sorted(set(ref) - set(got)):
        problems.append(f"missing metric: {key}")
    for key in sorted(set(got) - set(ref)):
        problems.append(f"unexpected metric: {key}")

    for key in sorted(set(ref) & set(got)):
        a, b = ref[key], got[key]
        if isinstance(a, float) or isinstance(b, float):
            if math.isclose(a, b, rel_tol=tol, abs_tol=tol):
                continue
        elif a == b:
            continue
        problems.append(f"{key}: expected {a}, got {b}")

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--tol", type=float, default=0.0)
    args = parser.parse_args()

    ref = load_metrics(args.reference)
    got = load_metrics(args.candidate)
    problems = compare(ref, got, args.tol)

    if not problems:
        print(f"[ok] {len(ref)} metrics match {args.reference.name}")
        return 0

    print(f"[FAIL] {len(problems)} difference(s) vs {args.reference.name}:")
    for line in problems:
        print(f"  {line}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
