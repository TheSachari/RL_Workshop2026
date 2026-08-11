"""Tests for the skill/role matching helpers.

These decide which firefighter can fill which role, so they drive the assignment
the whole simulation is built around.
"""

import gc

import numpy as np
import pandas as pd

from collective_functions import (
    _SKILL_WINDOW_CACHE,
    extract_skills,
    get_role_from_skills,
    get_skill_array,
    update_skills,
)


class TestGetRoleFromSkills:
    """Roles are matched against a firefighter's binary skill vector.

    `required_skills` is (n_roles, n_skills) with 1 = required, -1 = forbidden,
    0 = don't care. The result holds one entry per firefighter: the 1-based index
    of the first compatible role, or 0 when none fits.
    """

    def test_matches_a_required_skill(self):
        required = np.array([[1, 0, 0]])
        ff = np.array([[1, 0, 0]])
        assert get_role_from_skills(required, ff).tolist() == [1]

    def test_zero_when_no_role_fits(self):
        required = np.array([[1, 0, 0]])
        ff = np.array([[0, 0, 0]])  # lacks the required skill
        assert get_role_from_skills(required, ff).tolist() == [0]

    def test_selects_the_first_compatible_role(self):
        """Regression guard for the argmin/argmax slip.

        The reduction over the (n_roles, n_ff) compatibility matrix must be
        `argmax` — the first **compatible** role. It used to be `argmin`, which
        returns the first *incompatible* one, inverting the assignment whenever a
        vehicle function had more than one role (95% of them in the project data,
        and 31.8% of lookups gave a different answer).
        """
        required = np.array([[1, 0, 0], [0, 1, 0]])

        fits_role_zero = np.array([[1, 0, 0]])
        assert get_role_from_skills(required, fits_role_zero).tolist() == [1]

        fits_role_one = np.array([[0, 1, 0]])
        assert get_role_from_skills(required, fits_role_one).tolist() == [2]

    def test_picks_the_earliest_of_several_compatible_roles(self):
        """A firefighter fitting every role takes the first one."""
        required = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        assert get_role_from_skills(required, np.array([[1, 1, 1]])).tolist() == [1]

    def test_single_role_vehicles(self):
        required = np.array([[1, 0, 0]])
        assert get_role_from_skills(required, np.array([[1, 0, 0]])).tolist() == [1]
        assert get_role_from_skills(required, np.array([[0, 0, 0]])).tolist() == [0]

    def test_one_entry_per_firefighter(self):
        required = np.array([[1, 0, 0], [0, 1, 0]])
        ff = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        assert get_role_from_skills(required, ff).shape == (3,)

    def test_forbidden_skill_blocks_the_match(self):
        """-1 means the firefighter must NOT hold that skill."""
        required = np.array([[-1, 0, 0]])
        holds_it = np.array([[1, 0, 0]])
        lacks_it = np.array([[0, 0, 0]])
        assert get_role_from_skills(required, holds_it).tolist() == [0]
        assert get_role_from_skills(required, lacks_it).tolist() == [1]

    def test_zero_is_a_wildcard(self):
        """A 0 requirement accepts the skill present or absent."""
        required = np.array([[0, 0, 0]])
        for vector in ([[1, 1, 1]], [[0, 0, 0]], [[1, 0, 1]]):
            assert get_role_from_skills(required, np.array(vector)).tolist() == [1]

    def test_all_required_skills_must_be_held(self):
        required = np.array([[1, 1, 0]])
        partial = np.array([[1, 0, 0]])
        complete = np.array([[1, 1, 0]])
        assert get_role_from_skills(required, partial).tolist() == [0]
        assert get_role_from_skills(required, complete).tolist() == [1]


class TestGetSkillArray:
    """Builds a binary vector from `plus` (held) and `minus` (not held) skill names."""

    def make_df_skills(self):
        import pandas as pd
        return pd.DataFrame(columns=["INC2", "SAP1", "COD1"], index=[1, 2])

    def test_plus_skills_are_marked_present(self):
        df = self.make_df_skills()
        out = get_skill_array(["INC2"], [], df, zeros=3)
        assert out[0] == 1

    def test_minus_skills_are_marked_forbidden(self):
        df = self.make_df_skills()
        out = get_skill_array([], ["INC2"], df, zeros=3)
        assert out[0] == -1

    def test_unmentioned_skills_stay_neutral(self):
        df = self.make_df_skills()
        out = get_skill_array(["INC2"], [], df, zeros=3)
        assert out[1] == 0 and out[2] == 0

    def test_vector_length_follows_zeros(self):
        df = self.make_df_skills()
        assert len(get_skill_array(["INC2"], [], df, zeros=3)) == 3


class TestExtractSkills:
    """Parses the competence expressions stored in roles_competences.csv."""

    def test_single_skill(self):
        plus, minus, grade, sup = extract_skills("COD1")
        assert plus == ["COD1"]
        assert minus == []

    def test_plus_chains_requirements(self):
        plus, minus, _, _ = extract_skills("COD1 + INC2")
        assert plus == ["COD1", "INC2"]
        assert minus == []

    def test_minus_marks_an_exclusion(self):
        plus, minus, _, _ = extract_skills("COD1 - INC2")
        assert plus == ["COD1"]
        assert minus == ["INC2"]

    def test_plus_and_minus_combined(self):
        plus, minus, _, _ = extract_skills("COD1 + INC2 - SAP1")
        assert plus == ["COD1", "INC2"]
        assert minus == ["SAP1"]

    def test_no_grade_constraint_in_the_common_case(self):
        _, _, grade, sup = extract_skills("COD1 + INC2")
        assert grade == []
        assert sup == ""

    def test_grade_parsing_is_broken_but_unused(self):
        """GRADE(...) mis-parses: the [4:-1] slice drops the wrong characters.

        Pinned rather than fixed because no role expression in the project data
        uses GRADE (0 of 116), so changing it would alter nothing but could not
        be validated against real inputs.
        """
        _, _, grade, sup = extract_skills("COD1 + GRADE(SGT) > x")
        assert grade == "E(SGT"   # should be "SGT"
        assert sup == ">"


class TestSkillWindowCache:
    """The window cache must speed up repeat lookups without retaining tables.

    Both halves matter and they pull against each other. Keying on `id()` alone
    is unsound -- a collected temporary's address gets reused, and the next
    table of the same shape would read back another one's windows. Fixing that
    by storing the table itself leaks: `explainability.crew_skill_counts` builds
    a row-filtered view per scope, twice, and a year of evaluation touches ~60k
    scopes, so the cache pinned gigabytes of dead views and the run died of it.

    A weak reference satisfies both: dead entries cannot pass as hits, and
    nothing is kept alive.
    """

    @staticmethod
    def table(n_ff=40, n_skills=6):
        start, end = pd.Timestamp("2018-01-01"), pd.Timestamp("2030-01-01")
        columns = pd.MultiIndex.from_product(
            [[f"skill{i}" for i in range(n_skills)], ["Début", "Fin"]]
        )
        return pd.DataFrame(
            [[start, end] * n_skills for _ in range(n_ff)],
            index=range(n_ff), columns=columns,
        )

    def test_temporary_views_are_not_retained(self):
        """The regression: one entry after 200 throwaway views, not 200."""
        df = self.table()
        date = pd.Timestamp("2020-06-15")
        _SKILL_WINDOW_CACHE.clear()

        update_skills(df, date)                     # the long-lived table
        for i in range(200):
            update_skills(df.iloc[i % 20:(i % 20) + 10], date)
        gc.collect()

        assert len(_SKILL_WINDOW_CACHE) == 1

    def test_a_live_table_is_still_served_from_the_cache(self):
        df = self.table()
        date = pd.Timestamp("2020-06-15")
        _SKILL_WINDOW_CACHE.clear()

        first = update_skills(df, date)
        assert len(_SKILL_WINDOW_CACHE) == 1
        second = update_skills(df, date)
        assert np.array_equal(first, second)
        assert len(_SKILL_WINDOW_CACHE) == 1        # served, not recomputed

    def test_same_shaped_tables_do_not_share_windows(self):
        """What the identity check is for: equal shapes, different content."""
        date = pd.Timestamp("2020-06-15")
        _SKILL_WINDOW_CACHE.clear()

        valid = self.table(n_ff=4, n_skills=2)
        expired = self.table(n_ff=4, n_skills=2)
        # Same shape, but these windows closed before the reference date.
        expired.loc[:, :] = pd.Timestamp("2000-01-01")

        assert update_skills(valid, date).sum() == 8      # all valid
        assert update_skills(expired, date).sum() == 0    # none valid
