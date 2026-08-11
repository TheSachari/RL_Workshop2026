"""Per-decision records, for explaining individual assignments.

`dic_saved_skills` counts, over a whole run, how often the agent preserved a
rare skill. That is a global claim: it says the policy has a property, not why
it made any particular call. This module records the other half -- for one
assignment, which firefighters were actually assignable, what the agent thought
each was worth, and what would have been given up by choosing differently.

Everything here is a quantity the decision was already made from: the Q-values
`act` computes and then reduces to an argmax, the feasible set
`get_potential_actions` returns, and the rare skills `decide` already looks up.
Nothing is re-estimated after the fact, so a record is a description of the
decision rather than an approximation of it -- which is the part a post-hoc
attribution method cannot promise.

Sampling
--------
A ten-year evaluation makes several million decisions and almost all of them
are forced: one feasible firefighter, no choice to explain. Logging everything
would be both unaffordable and mostly empty, so records are kept when they carry
information:

* the choice was **close** -- the top two actions are within `margin`, so the
  agent was nearly indifferent and the assignment was arbitrary;
* a **rare skill was at stake** -- some feasible alternative carried a skill the
  next `n_hours` are expected to need;
* a **random sample** at `rate`, which keeps the log representative rather than
  a collection of edge cases.

The three reasons are recorded per entry, so an analysis can separate "decisions
that were hard" from "decisions that happened to be sampled".
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np

# Actions below this index designate a firefighter; 79 is "nobody assignable".
NO_ASSIGNMENT = 79


class DecisionLog:
    """Collects per-decision records and writes them as a pickle.

    Parameters
    ----------
    path :
        Destination file. Written atomically, and rewritten on each `flush`.
    rate :
        Probability of keeping a decision that is neither close nor
        skill-relevant. 0 disables the random sample.
    skills_of_interest :
        Restrict the rare-skill rule to these skill ids, or None for all of
        them. "Rare" is a threshold set when the dataset was built (`--rarity
        80`: fewer than 80 firefighters on duty hold the skill), and at that
        cut-off the rule fires on most decisions -- 78% of the test year --
        because it lumps together skills held by 79 people and skills held by
        three. Restricting it to the genuinely scarce ones is what makes a
        given-up skill worth reporting. `rarest_skills` computes such a set
        from a previous log.
    margin :
        Q-value gap below which a decision counts as close. Expressed in reward
        units, so it is comparable to the tariffs in the reward weights.

        The right value depends on how far the agent's Q-values have spread,
        which is a property of the trained weights, not of this code: on an
        untrained network they sit near zero and *every* decision reads as
        close. Check the margin distribution the first time you log a given
        agent -- `summary()` reports what fraction was kept -- and raise or
        lower this so "close" still selects a minority.
    keep_quantiles :
        Store the per-action return distribution as well as its mean. Costs
        `n_quantiles` floats per feasible action, so it is off by default.
    max_records :
        Stop collecting past this many records, so a long run cannot exhaust
        memory. Counting continues, and `summary()` reports what was skipped.
    seed :
        Seeds the sampling draw. Kept separate from the simulation's RNG so
        turning the log on cannot shift the run's own random stream.
    """

    def __init__(
        self,
        path,
        *,
        rate: float = 0.01,
        margin: float = 0.5,
        skills_of_interest=None,
        keep_quantiles: bool = False,
        max_records: int = 200_000,
        seed: int = 12345,
    ):
        self.path = Path(path)
        self.rate = rate
        self.margin = margin
        self.skills_of_interest = (
            None if skills_of_interest is None else set(int(s) for s in skills_of_interest)
        )
        self.keep_quantiles = keep_quantiles
        self.max_records = max_records
        self._rng = np.random.default_rng(seed)

        self.records: list[dict] = []
        self.seen = 0
        self.forced = 0
        self.dropped = 0

    def consider(
        self,
        *,
        num_inter,
        idx_role,
        num_d,
        current_station,
        v_mat,
        date,
        action,
        potential_actions,
        ff_existing,
        dic_rare_skills,
        upcoming_skills,
        skill_lvl,
        explain: Optional[dict] = None,
        local_rare_skills=None,
        irreversible_rare_skills=None,
    ) -> None:
        """Record this decision if it carries information; count it either way.

        `local_rare_skills` and `irreversible_rare_skills` are scoped to the
        station that would lose the firefighter: skills held by exactly one of
        its crew, and the subset of those the neighbouring stations cannot
        cover. They are recorded alongside the window-based `upcoming_skills`
        rather than replacing it -- one says a skill is about to be needed, the
        other says spending it here cannot be undone.
        """
        self.seen += 1

        feasible = [int(a) for a in potential_actions]
        # A single option, or no assignable firefighter at all: nothing was
        # decided, so there is nothing to explain.
        if len(feasible) < 2 or action == NO_ASSIGNMENT:
            self.forced += 1
            return

        q_values = (explain or {}).get("q_values")
        was_random = bool((explain or {}).get("random"))

        margin_value = None
        if q_values and not was_random and action in q_values:
            others = [q for a, q in q_values.items() if a != action]
            if others:
                margin_value = float(q_values[action] - max(others))

        # Which rare skills would each alternative have brought, and which of
        # those does the chosen firefighter not carry?
        upcoming = set(int(s) for s in np.asarray(upcoming_skills).ravel())
        if self.skills_of_interest is not None:
            upcoming &= self.skills_of_interest
        chosen_rare = set(int(s) for s in dic_rare_skills.get(action, ()))
        at_stake = {}
        for alt in feasible:
            if alt == action:
                continue
            alt_rare = set(int(s) for s in dic_rare_skills.get(alt, ()))
            given_up = sorted((alt_rare - chosen_rare) & upcoming)
            if given_up:
                at_stake[alt] = given_up

        local_rare = _as_id_set(local_rare_skills)
        irreversible = _as_id_set(irreversible_rare_skills)
        # Which of the chosen firefighter's skills nobody nearby can replace.
        chosen_skills = set(int(s) for s in dic_rare_skills.get(action, ()))
        spent_irreversibly = sorted(chosen_skills & irreversible)

        close = margin_value is not None and abs(margin_value) < self.margin
        skill_relevant = bool(at_stake)
        sampled = self.rate > 0 and self._rng.random() < self.rate

        if not (close or skill_relevant or sampled):
            return

        if len(self.records) >= self.max_records:
            self.dropped += 1
            return

        record = {
            "num_inter": int(num_inter),
            "date": date,
            "station": current_station,
            "vehicle": v_mat,
            "num_d": int(num_d),
            "idx_role": int(idx_role),
            "action": int(action),
            "ff_chosen": _ff_id(ff_existing, action),
            "feasible": feasible,
            "ff_feasible": {a: _ff_id(ff_existing, a) for a in feasible},
            "skill_lvl": float(skill_lvl),
            "q_values": q_values,
            "margin": margin_value,
            "exploratory": was_random,
            "rare_skills_upcoming": sorted(upcoming),
            "rare_skills_chosen": sorted(chosen_rare),
            "rare_skills_given_up": at_stake,
            # Scoped to the station losing the firefighter, unlike the three
            # above, which come from the event stream's department-wide column.
            "rare_skills_local": sorted(local_rare),
            "rare_skills_irreversible": sorted(irreversible),
            "irreversible_spent": spent_irreversibly,
            "why": {"close": close, "skill": skill_relevant, "sampled": sampled},
        }

        if self.keep_quantiles:
            quantiles = (explain or {}).get("quantiles")
            if quantiles:
                record["quantiles"] = {a: quantiles[a] for a in feasible if a in quantiles}

        self.records.append(record)

    def summary(self) -> dict:
        return {
            "decisions_seen": self.seen,
            "forced": self.forced,
            "recorded": len(self.records),
            "dropped_over_cap": self.dropped,
            "rate": self.rate,
            "margin": self.margin,
            "skills_of_interest": (
                None if self.skills_of_interest is None else sorted(self.skills_of_interest)
            ),
            # True once the cap bit: the records are then the *earliest*
            # eligible decisions, not a sample of them, and any rate computed
            # from them is biased towards the start of the run.
            "truncated": self.dropped > 0,
        }

    def flush(self) -> Path:
        """Write the log atomically, so an interrupted run keeps what it had."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"summary": self.summary(), "records": self.records}
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(payload, f)
        tmp.replace(self.path)
        return self.path


def rarest_skills(log_path, percentile: float = 5.0) -> set[int]:
    """The least-often-encountered skills in an existing log, as a set of ids.

    Counts how often each skill appeared as given up, and returns the bottom
    `percentile` by that count. Feed the result to a later run's
    `skills_of_interest` so its rare-skill rule fires only on the skills that
    are actually scarce.

    Frequency in the log, not the roster, is the right ranking here: a skill
    that never comes up in a decision cannot explain one, however few people
    hold it.
    """
    with open(log_path, "rb") as f:
        payload = pickle.load(f)

    counts: dict[int, int] = {}
    for record in payload["records"]:
        for skills in record.get("rare_skills_given_up", {}).values():
            for skill in skills:
                counts[int(skill)] = counts.get(int(skill), 0) + 1

    if not counts:
        return set()

    ranked = sorted(counts.items(), key=lambda kv: kv[1])
    keep = max(1, int(len(ranked) * percentile / 100))
    return {skill for skill, _ in ranked[:keep]}


def _as_id_set(skills) -> set:
    """Skill ids as a set, tolerating None and any array-like."""
    if skills is None:
        return set()
    return set(int(s) for s in np.asarray(skills).ravel())


def _ff_id(ff_existing, action):
    """Map an action index back to the firefighter it designates."""
    try:
        return int(ff_existing[action])
    except (IndexError, TypeError, ValueError):
        return None


def describe(record: dict) -> str:
    """Render one record as a sentence, for reading a log by eye."""
    who = record["ff_chosen"]
    lines = [
        f"Intervention {record['num_inter']} | {record['station']} | "
        f"vehicle {record['vehicle']} | role slot {record['idx_role']}",
        f"  chose firefighter {who} (action {record['action']}) "
        f"among {len(record['feasible'])} feasible",
    ]

    if record["exploratory"]:
        lines.append("  exploratory action: no Q-value behind this choice")
    elif record["q_values"]:
        ranked = sorted(record["q_values"].items(), key=lambda kv: -kv[1])[:3]
        shown = ", ".join(
            f"{record['ff_feasible'].get(a, a)}: {q:.3f}" for a, q in ranked
        )
        lines.append(f"  top Q-values -> {shown}")
        if record["margin"] is not None:
            lines.append(f"  margin over best alternative: {record['margin']:.3f}")

    if record["rare_skills_given_up"]:
        for alt, skills in record["rare_skills_given_up"].items():
            ff = record["ff_feasible"].get(alt, alt)
            lines.append(
                f"  not chosen: firefighter {ff} carried rare skill(s) {skills} "
                "that the coming window needs"
            )
    elif record["rare_skills_upcoming"]:
        lines.append("  no rare skill was given up by this choice")

    # Older logs predate the scoped fields, so read them defensively.
    if record.get("irreversible_spent"):
        lines.append(
            f"  committed skill(s) {record['irreversible_spent']}: sole holder at "
            "this station, and no neighbouring station can cover it"
        )

    return "\n".join(lines)
