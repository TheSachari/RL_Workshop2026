"""Tests for what counts as a rare skill at a given hour.

`rare_skills_required` is the 20th column of every event row and the input to
the whole explainability story, so its definition decides what the agent is
said to be preserving.

The rule tested here has a lower bound as well as an upper one, and the lower
one is easy to lose: `n < rarity` reads naturally and is wrong, because it makes
every skill nobody holds "rare". That version flagged 87% of skills per hour at
`--rarity 80`, and 61% even at `--rarity 5`, which drowned the signal in skills
no assignment could ever have traded.
"""

import numpy as np
import pandas as pd
import pytest

from explainability import get_rare_skills_from_planed_ff

STATION = "S1"
HOUR = (1, 1, 0)  # month, day, hour


def skills_table(holders_per_skill, n_ff=10):
    """A skills table where skill *i* is held by `holders_per_skill[i]` crew.

    Mirrors the real table's shape: a two-level column index of
    (skill, Début/Fin) with validity windows around the reference date.
    """
    start, end = pd.Timestamp("2018-01-01"), pd.Timestamp("2030-01-01")
    columns = pd.MultiIndex.from_product(
        [[f"skill{i}" for i in range(len(holders_per_skill))], ["Début", "Fin"]]
    )
    rows = []
    for ff in range(n_ff):
        row = []
        for held in holders_per_skill:
            # The first `held` firefighters carry this skill; the rest get NaT,
            # which `update_skills` reads as "not valid".
            row += [start, end] if ff < held else [pd.NaT, pd.NaT]
        rows.append(row)
    return pd.DataFrame(rows, index=range(n_ff), columns=columns)


def flagged(holders_per_skill, rarity, n_ff=10, top_k=None):
    df_skills = skills_table(holders_per_skill, n_ff)
    planning = {STATION: {HOUR[0]: {HOUR[1]: {HOUR[2]: {"planned": list(range(n_ff))}}}}}
    df_stations = pd.DataFrame({"Nom": [STATION]})
    result = get_rare_skills_from_planed_ff(
        pd.Timestamp("2020-06-15"), *HOUR, planning, df_stations, df_skills, rarity,
        top_k=top_k,
    )
    return set(result[0].tolist())


class TestLowerBound:
    def test_a_skill_nobody_holds_is_not_rare(self):
        """The bug this rule exists to prevent."""
        assert flagged([0, 0, 0], rarity=5) == set()

    def test_a_skill_one_person_holds_is_rare(self):
        assert flagged([1], rarity=5) == {0}

    def test_absent_and_scarce_are_told_apart(self):
        # skill 0: nobody. skill 1: one holder. skill 2: everyone.
        assert flagged([0, 1, 10], rarity=5) == {1}


class TestUpperBound:
    def test_a_skill_at_the_threshold_is_not_rare(self):
        """`< rarity`, so `rarity` holders is common enough."""
        assert flagged([5], rarity=5) == set()

    def test_a_skill_just_under_the_threshold_is_rare(self):
        assert flagged([4], rarity=5) == {0}

    @pytest.mark.parametrize("rarity,expected", [
        (2, {0}),           # only the single-holder skill
        (4, {0, 1}),        # 1 and 3 holders
        (8, {0, 1, 2}),     # 1, 3 and 7 holders
    ])
    def test_the_threshold_selects_progressively(self, rarity, expected):
        assert flagged([1, 3, 7, 0], rarity=rarity) == expected


class TestTopK:
    """The rank cut, which bounds how many skills an hour can flag.

    A count is not comparable across hours -- a day roster and a night roster
    make "fewer than N holders" mean different things -- so the threshold alone
    lets the number of flagged skills swing with the shift. The rank bounds it.
    """

    def test_only_the_k_rarest_survive(self):
        assert flagged([1, 2, 3, 4], rarity=10, top_k=2) == {0, 1}

    def test_the_threshold_still_applies_under_the_rank(self):
        """Asking for 3 does not promote a skill the threshold excluded."""
        # 2 holders passes; 8 and 9 are over `rarity`; 0 has nobody.
        assert flagged([2, 8, 9, 0], rarity=5, top_k=3) == {0}

    def test_fewer_eligible_than_k_returns_them_all(self):
        assert flagged([1, 2], rarity=10, top_k=20) == {0, 1}

    def test_none_keeps_the_threshold_alone(self):
        assert flagged([1, 2, 3, 4], rarity=10, top_k=None) == {0, 1, 2, 3}


class TestTies:
    """Ties are admitted whole, so the cut never splits equal counts.

    Taking exactly `top_k` would separate skills on equal counts by index order.
    Two skills held by 12 people are equally rare, and letting one in while the
    other stays out makes membership flip between neighbouring hours on nothing
    real -- noise that `get_related_rows_in_time` then spreads over a two-hour
    window and `rare_skills_given_up` reports per decision.
    """

    def test_a_tie_at_the_cutoff_is_kept_whole(self):
        """k=2, but three skills hold 2: all three stay rather than two of them."""
        assert flagged([2, 2, 2, 7], rarity=10, top_k=2) == {0, 1, 2}

    def test_the_band_widens_but_does_not_reach_past_the_cutoff_count(self):
        """Admitting a tie adds only skills *at* the cutoff, not the next ones up.

        k=3 and four skills hold 1: all four come in. Skill 4, at 5 holders, is
        strictly commoner than the cutoff and stays out even though the returned
        set is already over k.
        """
        assert flagged([1, 1, 1, 1, 5], rarity=10, top_k=3) == {0, 1, 2, 3}

    def test_membership_does_not_depend_on_skill_order(self):
        """The same multiset of counts, permuted, flags the same positions."""
        # Counts 3, 1, 3 are eligible; 9 is over `rarity`. k=2 cuts at 3, so the
        # pair of 3s comes in whole alongside the 1.
        assert flagged([3, 1, 3, 9], rarity=8, top_k=2) == {0, 1, 2}
        assert flagged([3, 3, 9, 1], rarity=8, top_k=2) == {0, 1, 3}

    def test_a_count_that_barely_moves_does_not_reshuffle_the_others(self):
        """The instability the tie rule exists to prevent, as a before/after."""
        before = flagged([4, 4, 4, 4, 9], rarity=8, top_k=3)
        after = flagged([4, 4, 4, 5, 9], rarity=8, top_k=3)
        assert before == {0, 1, 2, 3}
        # One holder joins skill 3, which leaves the band on its own count. The
        # other three are untouched rather than reshuffled around the cutoff.
        assert after == {0, 1, 2}


class TestCache:
    def test_a_cached_hour_returns_the_same_answer(self):
        df_skills = skills_table([1, 0, 10])
        planning = {STATION: {1: {1: {0: {"planned": list(range(10))}}}}}
        df_stations = pd.DataFrame({"Nom": [STATION]})
        cache = {}
        args = (pd.Timestamp("2020-06-15"), 1, 1, 0, planning, df_stations, df_skills, 5)

        first = get_rare_skills_from_planed_ff(*args, cache=cache)
        assert len(cache) == 1
        second = get_rare_skills_from_planed_ff(*args, cache=cache)
        assert np.array_equal(first[0], second[0])
        assert len(cache) == 1  # served from the cache, not recomputed
