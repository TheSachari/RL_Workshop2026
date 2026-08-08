"""Tests for the skill/role matching helpers.

These decide which firefighter can fill which role, so they drive the assignment
the whole simulation is built around.
"""

import numpy as np

from collective_functions import extract_skills, get_role_from_skills, get_skill_array


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

    def test_role_selection_is_inverted_by_an_argmin_argmax_slip(self):
        """BUG, pinned: with several roles, the wrong one is selected.

        The implementation reduces the (n_roles, n_ff) compatibility matrix with
        `np.argmin`, which returns the first **incompatible** role. It should be
        `np.argmax` (first compatible). Single-role vehicles and firefighters
        matching every role are unaffected, which is why it survives.

            firefighter fitting role 0 only -> reports 2 (should be 1)
            firefighter fitting role 1 only -> reports 1 (should be 2)

        This feeds gen_state via get_roles_for_ff, so it shapes the RL state on
        every decision. Left as-is deliberately: fixing it changes the agent's
        inputs and invalidates the recorded results, which is the author's call.
        """
        required = np.array([[1, 0, 0], [0, 1, 0]])

        fits_role_zero = np.array([[1, 0, 0]])
        assert get_role_from_skills(required, fits_role_zero).tolist() == [2]

        fits_role_one = np.array([[0, 1, 0]])
        assert get_role_from_skills(required, fits_role_one).tolist() == [1]

    def test_single_role_vehicles_are_unaffected_by_that_slip(self):
        """With one role, argmin and argmax agree — the common case still works."""
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
