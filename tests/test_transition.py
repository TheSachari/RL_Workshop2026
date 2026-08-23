"""Tests for `PendingTransition`, which owns the decide/learn one-step lag.

The behaviour these pin used to live as five keys in a shared `rl` dict, where
nothing enforced that `old_state` and `action` came from the same decision or
that the first decision emitted no transition.
"""

import pytest

from transition import PendingTransition, Transition


class TestLag:
    def test_first_close_emits_nothing(self):
        """The first decision has no predecessor -- the old `compute` guard."""
        p = PendingTransition(gamma=0.99)
        assert p.close("s0", done=False) is None

    def test_close_returns_the_previous_decision(self):
        p = PendingTransition(gamma=0.99)
        p.open("s0", action=3)
        p.add_reward(2.0)

        t = p.close("s1", done=False)

        assert t == Transition(state="s0", action=3, reward=2.0,
                               next_state="s1", done=False)

    def test_a_transition_is_emitted_once(self):
        """Closing twice must not replay the same transition."""
        p = PendingTransition(gamma=0.99)
        p.open("s0", action=1)
        assert p.close("s1", done=False) is not None
        assert p.close("s2", done=False) is None

    def test_is_open_tracks_the_pending_slot(self):
        p = PendingTransition(gamma=0.99)
        assert not p.is_open
        p.open("s0", action=0)
        assert p.is_open
        p.close("s1", done=False)
        assert not p.is_open

    def test_done_is_carried_through(self):
        p = PendingTransition(gamma=0.99)
        p.open("s0", action=0)
        assert p.close("s1", done=True).done is True


class TestReward:
    def test_rewards_accumulate(self):
        """Two callbacks for one decision sum; the old code kept only the last."""
        p = PendingTransition(gamma=0.99)
        p.open("s0", action=0)
        p.add_reward(1.5)
        p.add_reward(2.5)
        assert p.close("s1", done=False).reward == pytest.approx(4.0)

    def test_reward_without_a_pending_decision_is_dropped(self):
        p = PendingTransition(gamma=0.99)
        p.add_reward(99.0)          # no decision in flight
        p.open("s0", action=0)
        assert p.close("s1", done=False).reward == pytest.approx(0.0)

    def test_reward_does_not_leak_between_decisions(self):
        p = PendingTransition(gamma=0.99)
        p.open("s0", action=0)
        p.add_reward(5.0)
        p.close("s1", done=False)

        p.open("s1", action=1)      # no reward this time
        assert p.close("s2", done=False).reward == pytest.approx(0.0)


class TestShaping:
    def test_shaping_adds_the_potential_difference(self):
        p = PendingTransition(gamma=0.9, shaping_coeff=2.0)
        p.open("s0", action=0, potential=1.0)
        t = p.close("s1", done=False, potential=3.0)
        # 2.0 * (0.9 * 3.0 - 1.0)
        assert t.reward == pytest.approx(2.0 * (0.9 * 3.0 - 1.0))

    def test_shaping_off_by_default(self):
        p = PendingTransition(gamma=0.9)
        p.open("s0", action=0, potential=1.0)
        assert p.close("s1", done=False, potential=3.0).reward == pytest.approx(0.0)

    def test_missing_potential_skips_shaping(self):
        """Shaping enabled but potentials absent must not crash or bias."""
        p = PendingTransition(gamma=0.9, shaping_coeff=2.0)
        p.open("s0", action=0, potential=None)
        assert p.close("s1", done=False, potential=3.0).reward == pytest.approx(0.0)

        p.open("s1", action=0, potential=1.0)
        assert p.close("s2", done=False, potential=None).reward == pytest.approx(0.0)

    def test_shaping_telescopes_over_a_trajectory(self):
        """Ng et al. (1999): the shaping terms of a closed loop must cancel.

        Walking a cycle of potentials and summing only the shaping contribution
        leaves gamma-discounted residue, not accumulated bias -- the property
        that keeps the optimal policy unchanged.
        """
        gamma = 1.0                       # undiscounted makes the sum exact
        p = PendingTransition(gamma=gamma, shaping_coeff=1.0)
        potentials = [1.0, 4.0, 2.0, 1.0]  # returns to its start

        total = 0.0
        for prev, nxt in zip(potentials, potentials[1:]):
            p.open("s", action=0, potential=prev)
            total += p.close("s'", done=False, potential=nxt).reward

        assert total == pytest.approx(0.0)
