"""Tests for rarity scoped to the station that loses the firefighter.

The department-wide count these replace missed 89% of the skills that are scarce
where the decision is made: with ~39 skills held per station and a median crew of
17, a skill carried by 40 people across the department can still be the only one
on this shift.

Two scales, and the tests keep them apart because they answer different
questions -- `local` is "scarce here", `irreversible` is "and nobody nearby can
cover it". Roughly half of the sole-holder skills are absorbed by the
neighbourhood, so collapsing the two would report a trade-off that the
deployment order makes for free.
"""

import numpy as np
import pandas as pd

from explainability import (
    LOCAL_RARITY,
    NEIGH_RARITY,
    crew_skill_counts,
    rare_skills_for_step,
    scoped_rare_skills,
    station_neighbourhood,
)
from tests.test_rarity import skills_table


class FakeStep:
    """The handful of `_LoopState` fields `rare_skills_for_step` reads."""

    def __init__(self, station, pdd, planning, df_skills, dic_station_distance=None):
        self.current_station = station
        self.pdd = pdd
        self.planning = planning
        self.df_skills = df_skills
        self.dic_station_distance = dic_station_distance or {}
        self.date = pd.Timestamp("2020-06-15")
        self.month, self.day, self.hour = 1, 1, 0


class TestScopedRareSkills:
    def test_a_sole_holder_scarce_around_is_irreversible(self):
        local, irrev = scoped_rare_skills(np.array([1]), np.array([3]))
        assert local.tolist() == [0]
        assert irrev.tolist() == [0]

    def test_a_sole_holder_the_neighbourhood_covers_is_absorbable(self):
        """Scarce here, plentiful next door: the departure order handles it."""
        local, irrev = scoped_rare_skills(np.array([1]), np.array([40]))
        assert local.tolist() == [0]
        assert irrev.tolist() == []

    def test_two_holders_here_is_not_scarce_at_all(self):
        """`LOCAL_RARITY` is the sole holder; from two on, one remains."""
        local, irrev = scoped_rare_skills(np.array([2]), np.array([2]))
        assert local.tolist() == []
        assert irrev.tolist() == []

    def test_a_skill_nobody_here_holds_is_not_a_trade_off(self):
        """Even when the neighbourhood is out of it too -- there is no one to assign."""
        local, irrev = scoped_rare_skills(np.array([0]), np.array([0]))
        assert local.tolist() == []
        assert irrev.tolist() == []

    def test_irreversible_is_a_subset_of_local(self):
        local_counts = np.array([1, 1, 1, 5, 0])
        neigh_counts = np.array([2, 30, 4, 1, 0])
        local, irrev = scoped_rare_skills(local_counts, neigh_counts)
        assert local.tolist() == [0, 1, 2]      # 3 has two holders, 4 none
        assert irrev.tolist() == [0, 2]         # 1 is covered next door
        assert set(irrev).issubset(set(local))

    def test_a_skill_absent_from_the_neighbourhood_count_is_not_irreversible(self):
        """Zero around means the neighbourhood cannot supply it either.

        The lower bound is deliberate and mirrors the local one: `neigh > 0`.
        A skill the neighbourhood does not hold at all is a gap in the sector,
        not something this assignment spends -- the same reasoning that keeps
        never-held skills out of the local count.
        """
        local, irrev = scoped_rare_skills(np.array([1]), np.array([0]))
        assert local.tolist() == [0]
        assert irrev.tolist() == []

    def test_thresholds_are_overridable(self):
        counts = np.array([3])
        assert scoped_rare_skills(counts, np.array([3]))[0].tolist() == []
        assert scoped_rare_skills(counts, np.array([3]), local_rarity=4)[0].tolist() == [0]

    def test_the_defaults_are_the_measured_ones(self):
        assert (LOCAL_RARITY, NEIGH_RARITY) == (2, 6)


class TestStationNeighbourhood:
    """Always the neighbourhood of the station that loses the crew."""

    PDD = ["A", "B", "C", "D", "E", "F", "G", "H"]
    DIST = {"A": {"Z": 3, "Y": 7, "X": 9}, "B": {"A": 2}}

    def test_a_departure_follows_the_pdd_order(self):
        """The order the departure will actually work through, not distance."""
        assert station_neighbourhood("A", self.PDD, self.DIST, 3) == ["B", "C", "D"]

    def test_a_station_mid_pdd_looks_at_what_follows_it(self):
        assert station_neighbourhood("C", self.PDD, self.DIST, 2) == ["D", "E"]

    def test_the_last_pdd_station_has_no_followers(self):
        assert station_neighbourhood("H", self.PDD, self.DIST, 3) == []

    def test_a_reinforcement_falls_back_to_distance(self):
        """No PDD: the sender's own nearest stations stand in.

        Reinforcements are not skipped. The crew leaves the sending station and
        stays away for the travel time plus 20 minutes with no local incident,
        and a sender's roster is thinner than a first-due station's, so this is
        where spending the last holder costs most.
        """
        assert station_neighbourhood("A", [], self.DIST, 2) == ["Z", "Y"]

    def test_a_station_absent_from_the_pdd_falls_back_to_distance(self):
        assert station_neighbourhood("A", ["P", "Q"], self.DIST, 2) == ["Z", "Y"]

    def test_the_station_never_neighbours_itself(self):
        dist = {"A": {"A": 0, "Z": 3}}
        assert station_neighbourhood("A", [], dist, 2) == ["Z"]

    def test_an_unknown_station_yields_no_neighbourhood(self):
        assert station_neighbourhood("UNKNOWN", [], self.DIST, 3) == []

    def test_no_distance_table_yields_no_neighbourhood(self):
        assert station_neighbourhood("A", [], None, 3) == []


class TestCrewSkillCounts:
    DATE = pd.Timestamp("2020-06-15")

    def planning_for(self, crews):
        return {name: {1: {1: {0: {"planned": crew}}}} for name, crew in crews.items()}

    def test_it_counts_holders_across_the_given_stations(self):
        # 6 firefighters; skill 0 held by the first 2, skill 1 by the first 5.
        df_skills = skills_table([2, 5], n_ff=6)
        planning = self.planning_for({"S1": [0, 1, 2], "S2": [3, 4, 5]})

        both = crew_skill_counts(["S1", "S2"], planning, 1, 1, 0, df_skills, self.DATE)
        assert both.tolist() == [2, 5]

        # S1 alone holds firefighters 0-2: both of skill 0's holders, three of
        # skill 1's -- so the same skill is scarce or not depending on scope.
        one = crew_skill_counts(["S1"], planning, 1, 1, 0, df_skills, self.DATE)
        assert one.tolist() == [2, 3]

    def test_an_empty_roster_counts_zero_rather_than_failing(self):
        df_skills = skills_table([2, 5], n_ff=6)
        counts = crew_skill_counts([], self.planning_for({}), 1, 1, 0, df_skills, self.DATE)
        assert counts.tolist() == [0, 0]

    def test_a_station_missing_from_the_planning_is_skipped(self):
        """A neighbour with no entry for this hour must not abort the count."""
        df_skills = skills_table([2, 5], n_ff=6)
        planning = self.planning_for({"S1": [0, 1, 2]})
        counts = crew_skill_counts(
            ["S1", "ABSENT"], planning, 1, 1, 0, df_skills, self.DATE
        )
        assert counts.tolist() == [2, 3]

    def test_a_matricule_absent_from_the_skills_table_is_skipped(self):
        df_skills = skills_table([2, 5], n_ff=6)
        planning = self.planning_for({"S1": [0, 1, 999]})
        counts = crew_skill_counts(["S1"], planning, 1, 1, 0, df_skills, self.DATE)
        assert counts.tolist() == [2, 2]


class TestRareSkillsForStep:
    """The decision-time entry point, reading scope off the loop state."""

    def build(self, station, pdd, crews, holders, n_ff, dist=None):
        df_skills = skills_table(holders, n_ff=n_ff)
        planning = {name: {1: {1: {0: {"planned": crew}}}} for name, crew in crews.items()}
        return FakeStep(station, pdd, planning, df_skills, dist)

    def test_a_departure_scopes_to_the_station_and_its_pdd_followers(self):
        # Skill 0 held only by ff 0 (at S1); skill 1 held by ff 0-7 (spread).
        st = self.build(
            "S1", ["S1", "S2"],
            {"S1": [0, 1, 2], "S2": [3, 4, 5, 6, 7]},
            holders=[1, 8], n_ff=8,
        )
        local, irrev = rare_skills_for_step(st)
        # Skill 0: sole holder at S1, and the neighbourhood has no other -> irreversible.
        assert local.tolist() == [0]
        assert irrev.tolist() == [0]

    def test_a_neighbour_holding_the_skill_makes_it_absorbable(self):
        # Skill 0 held by ff 0-6: one at S1, six at S2 -- plenty next door.
        st = self.build(
            "S1", ["S1", "S2"],
            {"S1": [0], "S2": [1, 2, 3, 4, 5, 6]},
            holders=[7], n_ff=7,
        )
        local, irrev = rare_skills_for_step(st)
        assert local.tolist() == [0]     # still the sole holder at S1
        assert irrev.tolist() == []      # but S2 has six

    def test_a_reinforcement_uses_the_senders_own_neighbourhood(self):
        """No PDD, so the sender's nearest stations stand in -- not the destination's."""
        st = self.build(
            "SENDER", [],
            {"SENDER": [0], "NEAR": [1, 2, 3, 4, 5, 6], "FAR": [7]},
            holders=[7], n_ff=8,
            dist={"SENDER": {"NEAR": 4, "FAR": 30}},
        )
        local, irrev = rare_skills_for_step(st, n_following=1)
        # NEAR is the one neighbour, and it holds six: absorbable.
        assert local.tolist() == [0]
        assert irrev.tolist() == []

    def test_a_reinforcement_with_a_thin_neighbourhood_is_irreversible(self):
        st = self.build(
            "SENDER", [],
            {"SENDER": [0], "NEAR": [7], "FAR": [1, 2, 3, 4, 5, 6]},
            holders=[7], n_ff=8,
            dist={"SENDER": {"NEAR": 4, "FAR": 30}},
        )
        local, irrev = rare_skills_for_step(st, n_following=1)
        # Only NEAR counts, and it does not hold the skill. FAR does, but it is
        # outside the scope -- too far to cover the gap this sending opens.
        assert local.tolist() == [0]
        assert irrev.tolist() == [0]

    def test_the_scope_changes_the_answer(self):
        """The same hour, two stations, two different verdicts.

        This is what the department-wide count could not express: it produced
        one array per event, so both stations got the same answer.
        """
        crews = {"S1": [0], "S2": [1, 2, 3, 4, 5, 6]}
        holders = [7]  # skill 0 held by all seven
        a = self.build("S1", ["S1"], crews, holders, n_ff=7)
        b = self.build("S2", ["S2"], crews, holders, n_ff=7)
        assert rare_skills_for_step(a)[0].tolist() == [0]   # sole holder at S1
        assert rare_skills_for_step(b)[0].tolist() == []    # six at S2

    def test_the_cache_serves_a_repeated_scope(self):
        """Every role of every vehicle at a station shares one entry."""
        st = self.build("S1", ["S1", "S2"], {"S1": [0], "S2": [1]}, [2], n_ff=2)
        cache = {}
        first = rare_skills_for_step(st, cache=cache)
        assert len(cache) == 1
        second = rare_skills_for_step(st, cache=cache)
        assert len(cache) == 1
        assert np.array_equal(first[0], second[0])
        assert np.array_equal(first[1], second[1])

    def test_the_cache_separates_stations_at_the_same_hour(self):
        """Keying on the hour alone would hand S2 the answer computed for S1."""
        crews = {"S1": [0], "S2": [1, 2, 3, 4, 5, 6]}
        cache = {}
        a = self.build("S1", ["S1"], crews, [7], n_ff=7)
        b = self.build("S2", ["S2"], crews, [7], n_ff=7)
        assert rare_skills_for_step(a, cache=cache)[0].tolist() == [0]
        assert rare_skills_for_step(b, cache=cache)[0].tolist() == []
        assert len(cache) == 2

    def test_the_cache_separates_differing_neighbourhoods(self):
        """Same station, different PDD: the neighbourhood scope differs."""
        crews = {"S1": [0], "RICH": [1, 2, 3, 4, 5, 6], "POOR": [7]}
        cache = {}
        rich = self.build("S1", ["S1", "RICH"], crews, [7], n_ff=8)
        poor = self.build("S1", ["S1", "POOR"], crews, [7], n_ff=8)
        assert rare_skills_for_step(rich, cache=cache)[1].tolist() == []
        assert rare_skills_for_step(poor, cache=cache)[1].tolist() == [0]
        assert len(cache) == 2


class TestTheTwoScalesTogether:
    """The end-to-end shape: counts at two scopes, then the split."""

    DATE = pd.Timestamp("2020-06-15")

    def test_a_sole_holder_locally_is_absorbed_by_a_rich_neighbourhood(self):
        # Skill 0: held by 1 of the 12. Skill 1: held by all 12.
        df_skills = skills_table([1, 12], n_ff=12)
        planning = {
            "S1": {1: {1: {0: {"planned": [0, 1, 2]}}}},      # holds skill 0
            "S2": {1: {1: {0: {"planned": list(range(3, 12))}}}},
        }
        local = crew_skill_counts(["S1"], planning, 1, 1, 0, df_skills, self.DATE)
        neigh = crew_skill_counts(["S1", "S2"], planning, 1, 1, 0, df_skills, self.DATE)

        assert local.tolist() == [1, 3]     # sole holder of skill 0
        assert neigh.tolist() == [1, 12]

        rare, irrev = scoped_rare_skills(local, neigh)
        assert rare.tolist() == [0]
        # Only one holder in the whole neighbourhood too: nobody can cover it.
        assert irrev.tolist() == [0]
