"""Summarise a decision log, and compare a run's metrics against a baseline.

    python analyse_decisions.py --log decisions_eval.pkl
    python analyse_decisions.py --log decisions_eval.pkl \
        --metrics metrics_agent_eval.pkl --baseline metrics_heuristic.pkl
    python analyse_decisions.py --log decisions_eval.pkl --examples 5 --sort margin

Reads from `Plots/` unless given a path. Prints; writes nothing.

What this does and does not claim
--------------------------------
The log holds the quantities the agent decided from, so counting them is
descriptive: "the choice was forced 62% of the time" is a fact about the run.

Two things it cannot tell you, and which this script therefore refuses to
print. First, whether a decision was *good*: the log has no counterfactual
outcome, only the alternative the agent declined, and a rare skill given up may
well have been the right call. Second, whether a Q-value gap is *large*: Q is on
the reward's scale, which the tariffs set arbitrarily, so margins are
comparable across decisions of one agent and meaningless across agents.

The metric comparison is likewise a difference between two runs, not an
attribution: the agent and the heuristic differ in every decision they make, so
a gap in `rupture_ff` is an outcome, not evidence about any particular choice.
"""

from __future__ import annotations

import argparse
import pickle

import numpy as np

from decision_log import describe
from paths import PLOTS, resolve

# Indicators where a smaller value is the better outcome. The rest (crews sent,
# skill level) are context, not a score, so they are reported without a verdict.
LOWER_IS_BETTER = {
    "rupture_ff", "v_degraded", "function_not_found", "v1_not_sent_from_s1",
    "v3_not_sent_from_s3", "v_not_found_in_last_station",
}


def load(path_like, default_dir=None):
    path = resolve(path_like, default_dir or PLOTS)
    if not path.is_file():
        raise SystemExit(f"not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f), path


def pct(n, d):
    return f"{100 * n / d:.1f}%" if d else "n/a"


def report_coverage(summary):
    seen = summary["decisions_seen"]
    forced = summary["forced"]
    print("Decisions")
    print(f"  seen                : {seen}")
    print(f"  forced (no choice)  : {forced}  ({pct(forced, seen)})")
    print(f"  genuine choices     : {seen - forced}  ({pct(seen - forced, seen)})")
    print(f"  recorded            : {summary['recorded']}")
    if summary.get("dropped_over_cap"):
        print(f"  dropped over cap    : {summary['dropped_over_cap']}")
        print("  TRUNCATED -- the cap was reached, so these records are the")
        print("  earliest eligible decisions, not a sample of them. Proportions")
        print("  below are biased towards the start of the run. Re-run with a")
        print("  lower --decision_log_skill_rate or a higher cap.")
    of_interest = summary.get("skills_of_interest")
    if of_interest is not None:
        print(f"  rare-skill rule limited to {len(of_interest)} skills: {of_interest}")
        print("  Counts below cover only those; they are not run-wide totals.")
    print("  A forced decision has one feasible firefighter: the assignment is")
    print("  made by the skill constraints, not by the policy.")


def report_margins(records):
    """The margin distribution, which is also how you calibrate the log."""
    margins = [r["margin"] for r in records if r.get("margin") is not None]
    exploratory = sum(1 for r in records if r.get("exploratory"))

    print("\nMargin over the best alternative")
    if exploratory:
        print(f"  exploratory actions : {exploratory}  (no Q behind them; excluded)")
    if not margins:
        print("  no Q-values in this log -- was it recorded during training?")
        return

    m = np.asarray(margins)
    qs = np.percentile(m, [50, 90, 99])
    print(f"  n                   : {len(m)}")
    print(f"  min / median / max  : {m.min():.4f} / {qs[0]:.4f} / {m.max():.4f}")
    print(f"  p90 / p99           : {qs[1]:.4f} / {qs[2]:.4f}")
    near_zero = int((m < 0.01).sum())
    print(f"  under 0.01          : {near_zero}  ({pct(near_zero, len(m))})")
    print("  Margins are on the reward's scale. Use this spread to set")
    print("  --decision_log_margin so 'close' stays a minority for THIS agent;")
    print("  the values do not transfer between agents.")


def report_rare_skills(records):
    """How often a rejected alternative carried a skill the window needs."""
    at_stake = [r for r in records if r.get("rare_skills_given_up")]
    print("\nRare skills")
    print(f"  records where an alternative carried a needed rare skill: {len(at_stake)}")
    if not at_stake:
        return

    per_skill: dict[int, int] = {}
    for r in at_stake:
        for skills in r["rare_skills_given_up"].values():
            for s in skills:
                per_skill[s] = per_skill.get(s, 0) + 1

    top = sorted(per_skill.items(), key=lambda kv: -kv[1])[:10]
    print(f"  distinct skills involved : {len(per_skill)}")
    print("  most often given up      : " +
          ", ".join(f"{s} ({n}x)" for s, n in top))

    # Was the agent paying for it, or indifferent? Descriptive only.
    with_margin = [r["margin"] for r in at_stake if r.get("margin") is not None]
    if with_margin:
        print(f"  median margin on those   : {np.median(with_margin):.4f}")
        print("  A skill given up is not an error: the agent may have had good")
        print("  reason. The log records the trade-off, it does not score it.")


def report_examples(records, n, sort):
    if n <= 0 or not records:
        return
    pool = [r for r in records if r.get("rare_skills_given_up")] or records
    if sort == "margin":
        pool = sorted((r for r in pool if r.get("margin") is not None),
                      key=lambda r: r["margin"])
    print(f"\nExamples ({min(n, len(pool))} of {len(pool)})")
    for r in pool[:n]:
        print()
        print(describe(r))


def comparable(metrics, baseline):
    """Whether the two runs covered enough of the same work to be compared.

    Every indicator is a count accumulated over a run, so two runs of different
    lengths differ on all of them for a reason that has nothing to do with the
    policy. `v_required` -- vehicles the incidents asked for -- stands in for
    the workload each run faced.
    """
    a, b = metrics.get("v_required"), baseline.get("v_required")
    if not a or not b:
        return True, ""
    ratio = a / b
    if 0.9 <= ratio <= 1.1:
        return True, ""
    return False, (
        f"workloads differ: v_required {a} vs {b} ({ratio:.2f}x). These are"
        " cumulative counts, so the deltas below mostly measure run length."
    )


def report_metrics(metrics, baseline):
    print("\nOutcome metrics")
    keys = sorted(set(metrics) | set(baseline or {}))
    width = max(len(k) for k in keys)

    if baseline is None:
        for k in keys:
            print(f"  {k:<{width}} : {metrics[k]}")
        return

    ok, why = comparable(metrics, baseline)
    if not ok:
        print(f"  NOT COMPARABLE -- {why}")
        print("  Verdicts are withheld. Re-run both over the same --start/--end.")

    print(f"  {'':<{width}}   {'agent':>12} {'baseline':>12} {'delta':>12}")
    for k in keys:
        a, b = metrics.get(k), baseline.get(k)
        if a is None or b is None:
            print(f"  {k:<{width}} : {str(a):>12} {str(b):>12} {'--':>12}")
            continue
        delta = a - b
        mark = ""
        if ok and k in LOWER_IS_BETTER and delta != 0:
            mark = "  better" if delta < 0 else "  worse"
        print(f"  {k:<{width}} : {a:>12} {b:>12} {delta:>+12}{mark}")

    if ok:
        print("  'better'/'worse' compares two whole runs on the indicators where a")
        print("  smaller count is preferable. It attributes nothing to any single")
        print("  decision, and says nothing about statistical significance.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", required=True, help="decision log written by --decision_log")
    parser.add_argument("--metrics", default=None, help="run metrics (dic_indic) to report")
    parser.add_argument("--baseline", default=None, help="metrics to compare against")
    parser.add_argument("--examples", type=int, default=3, help="decisions to print in full")
    parser.add_argument("--sort", choices=("none", "margin"), default="none",
                        help="'margin' shows the closest calls first")
    args = parser.parse_args()

    payload, path = load(args.log)
    print(f"Decision log: {path}")
    report_coverage(payload["summary"])
    report_margins(payload["records"])
    report_rare_skills(payload["records"])
    report_examples(payload["records"], args.examples, args.sort)

    if args.metrics:
        metrics, _ = load(args.metrics)
        baseline = load(args.baseline)[0] if args.baseline else None
        report_metrics(metrics, baseline)


if __name__ == "__main__":
    main()
