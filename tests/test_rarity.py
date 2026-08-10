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


def flagged(holders_per_skill, rarity, n_ff=10):
    df_skills = skills_table(holders_per_skill, n_ff)
    planning = {STATION: {HOUR[0]: {HOUR[1]: {HOUR[2]: {"planned": list(range(n_ff))}}}}}
    df_stations = pd.DataFrame({"Nom": [STATION]})
    result = get_rare_skills_from_planed_ff(
        pd.Timestamp("2020-06-15"), *HOUR, planning, df_stations, df_skills, rarity
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
