"""Tests for the action-selection path: `get_potential_actions` and `apply_logic`.

Both simulation arms now go through these (the heuristic directly, the agent to
build its feasible-action mask), so a change here moves the baseline *and* the
agent at once — which is exactly the kind of shift a whole-run comparison cannot
attribute.
"""

import random

import numpy as np
import pytest

from collective_functions import apply_logic, get_potential_actions

N_FF = 6          # firefighter rows in the toy state
N_COLS = 10       # feature columns
ROLE_COL = 3      # column standing for "the current role"

# The last three columns encode a firefighter's unavailability; get_potential_actions
# requires them to be all-zero for a firefighter to be selectable.
BUSY_COLS = slice(-3, None)


def make_state(skills, busy=(), role_col=ROLE_COL):
    """Build a state matrix in the layout `gen_state` produces.

    row 0    : global RL info (unused here)
    row 1    : one-hot current role
    rows 2.. : one row per firefighter; `role_col` holds the skill level
    """
    state = np.zeros((2 + N_FF, N_COLS))
    state[1, role_col] = 1
    for i, lvl in enumerate(skills):
        state[2 + i, role_col] = lvl
    for i in busy:
        state[2 + i, BUSY_COLS] = 1
    return state


class TestGetPotentialActions:
    def test_returns_firefighters_with_a_positive_skill(self):
        state = make_state([0, 2, 0, 5, 0, 0])
        actions, skills = get_potential_actions(state, all_ff_waiting=False)
        assert actions == [1, 3]
        assert skills == [2.0, 5.0]

    def test_ordering_follows_firefighter_index(self):
        state = make_state([9, 1, 4, 0, 0, 0])
        actions, skills = get_potential_actions(state, all_ff_waiting=False)
        assert actions == [0, 1, 2]
        assert skills == [9.0, 1.0, 4.0]

    def test_busy_firefighters_are_excluded(self):
        """A positive skill is not enough; the trailing columns must be clear."""
        state = make_state([3, 3, 3, 0, 0, 0], busy=[1])
        actions, skills = get_potential_actions(state, all_ff_waiting=False)
        assert actions == [0, 2]
        assert skills == [3.0, 3.0]

    def test_no_feasible_firefighter_returns_the_79_sentinel(self):
        state = make_state([0] * N_FF)
        actions, skills = get_potential_actions(state, all_ff_waiting=False)
        assert actions == [79]
        assert skills == [0]

    def test_sentinel_asserts_the_last_slot_is_empty(self):
        """The 79 sentinel doubles as a slot-79 collision check."""
        state = make_state([0] * N_FF)
        state[-1, 0] = 1  # last firefighter row no longer all-zero
        with pytest.raises(AssertionError, match="slot 79"):
            get_potential_actions(state, all_ff_waiting=False)

    def test_all_ff_waiting_picks_the_first_marked_firefighter(self):
        """In waiting mode the choice is forced: first firefighter flagged in col -2."""
        state = make_state([0] * N_FF)
        state[2, -2] = 1  # firefighter 0
        state[4, -2] = 1  # firefighter 2, must be ignored
        actions, _ = get_potential_actions(state, all_ff_waiting=True)
        assert actions == [0]

    def test_all_ff_waiting_requires_action_zero(self):
        """If the first flagged firefighter is not index 0, the invariant fails."""
        state = make_state([0] * N_FF)
        state[3, -2] = 1  # firefighter 1, not 0
        with pytest.raises(AssertionError, match="all_ff_waiting"):
            get_potential_actions(state, all_ff_waiting=True)


class TestApplyLogic:
    def test_best_picks_the_lowest_skill_level(self):
        """Lower skill level means a tighter match, so `is_best` minimises it."""
        action, skill = apply_logic([4, 7, 9], [3, 1, 2], is_best=True)
        assert (action, skill) == (7, 1)

    def test_best_is_deterministic(self):
        for _ in range(20):
            assert apply_logic([4, 7, 9], [3, 1, 2], is_best=True) == (7, 1)

    def test_best_takes_the_first_of_equal_minima(self):
        action, skill = apply_logic([4, 7, 9], [1, 1, 5], is_best=True)
        assert (action, skill) == (4, 1)

    def test_random_returns_a_feasible_action_with_its_own_skill(self):
        actions, skills = [4, 7, 9], [3, 1, 2]
        pairs = dict(zip(actions, skills))
        for seed in range(25):
            random.seed(seed)
            action, skill = apply_logic(actions, skills, is_best=False)
            assert action in actions
            assert skill == pairs[action]

    def test_random_is_seed_reproducible(self):
        random.seed(7)
        first = apply_logic([4, 7, 9], [3, 1, 2], is_best=False)
        random.seed(7)
        assert apply_logic([4, 7, 9], [3, 1, 2], is_best=False) == first

    def test_random_actually_varies(self):
        """Guards against the random branch silently collapsing to a constant."""
        seen = set()
        for seed in range(40):
            random.seed(seed)
            seen.add(apply_logic([4, 7, 9], [3, 1, 2], is_best=False)[0])
        assert len(seen) > 1

    def test_single_candidate_is_returned_by_both_modes(self):
        assert apply_logic([79], [0], is_best=True) == (79, 0)
        assert apply_logic([79], [0], is_best=False) == (79, 0)
