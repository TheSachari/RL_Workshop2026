"""Tests for `compute_reward`.

This is the function the agent optimises, so a silent change here retrains the
agent towards a different objective. The golden cases would notice only if the
resulting metrics moved implausibly — they pin whole-run behaviour, not the
reward contract itself.
"""

import pytest

from collective_functions import compute_reward

# The indicator set the simulation actually carries (see load_environment_variables).
INDICATORS = [
    "v_required", "v_sent", "v_sent_full", "v_degraded", "rupture_ff",
    "function_not_found", "v1_not_sent_from_s1", "v3_not_sent_from_s3",
    "v_not_found_in_last_station", "ff_required", "ff_sent",
    "z1_VSAV_sent", "z1_FPT_sent", "z1_EPA_sent",
    "VSAV_needed", "FPT_needed", "EPA_needed",
    "VSAV_disp", "FPT_disp", "EPA_disp", "skill_lvl",
]

# `disp` availability high enough that none of the three bonuses fire,
# isolating the delta term.
STOCKED = {"VSAV_disp": 5, "FPT_disp": 5, "EPA_disp": 5}


def indicators(**overrides):
    d = {k: 0 for k in INDICATORS}
    d.update(STOCKED)
    d.update(overrides)
    return d


def tariffs(**overrides):
    d = {k: 0 for k in INDICATORS}
    d.update(overrides)
    return d


def test_no_change_gives_zero_reward():
    state = indicators()
    assert compute_reward(state, dict(state), 1, tariffs()) == 0


def test_reward_is_the_weighted_delta():
    old = indicators()
    new = indicators(rupture_ff=3)
    reward = compute_reward(new, old, 1, tariffs(rupture_ff=-100))
    assert reward == -300


def test_deltas_accumulate_across_indicators():
    old = indicators()
    new = indicators(v_sent=2, v_degraded=1)
    reward = compute_reward(new, old, 1, tariffs(v_sent=10, v_degraded=-100))
    assert reward == 2 * 10 + 1 * -100


def test_negative_delta_flips_the_sign():
    """Indicators can decrease; the weighting must follow."""
    old = indicators(v_sent=5)
    new = indicators(v_sent=3)
    assert compute_reward(new, old, 1, tariffs(v_sent=10)) == -20


@pytest.mark.parametrize("num_d", [79, 80, 99, 100, 101])
def test_reinforcement_steps_score_nothing(num_d):
    """num_d >= 79 marks a reinforcement rather than a decision: reward stays 0.

    99/100/101 are the VSAV/FPT/EPA reinforcement sentinels.
    """
    old = indicators()
    new = indicators(rupture_ff=10)
    assert compute_reward(new, old, num_d, tariffs(rupture_ff=-100)) == 0


@pytest.mark.parametrize("num_d", [0, 1, 5, 78])
def test_regular_steps_do_score(num_d):
    old = indicators()
    new = indicators(rupture_ff=1)
    assert compute_reward(new, old, num_d, tariffs(rupture_ff=-100)) == -100


class TestAvailabilityBonus:
    """The three `_disp` indicators bypass the delta and act as level thresholds."""

    def test_vsav_bonus_applies_below_two(self):
        state = indicators(VSAV_disp=1)
        assert compute_reward(state, dict(state), 1, tariffs(VSAV_disp=-50)) == -50

    def test_vsav_bonus_absent_at_two(self):
        state = indicators(VSAV_disp=2)
        assert compute_reward(state, dict(state), 1, tariffs(VSAV_disp=-50)) == 0

    def test_epa_threshold_is_one_not_two(self):
        """EPA is scarcer: its threshold is < 1, unlike VSAV/FPT which use < 2."""
        at_one = indicators(EPA_disp=1)
        assert compute_reward(at_one, dict(at_one), 1, tariffs(EPA_disp=-50)) == 0

        at_zero = indicators(EPA_disp=0)
        assert compute_reward(at_zero, dict(at_zero), 1, tariffs(EPA_disp=-50)) == -50

    def test_disp_indicators_are_excluded_from_the_delta(self):
        """A change in `_disp` must not be counted twice (delta *and* threshold)."""
        old = indicators(VSAV_disp=5)
        new = indicators(VSAV_disp=4)  # still >= 2, so no bonus either
        assert compute_reward(new, old, 1, tariffs(VSAV_disp=-50)) == 0

    def test_all_three_bonuses_stack(self):
        state = indicators(VSAV_disp=0, FPT_disp=0, EPA_disp=0)
        reward = compute_reward(
            state, dict(state), 1,
            tariffs(VSAV_disp=-10, FPT_disp=-20, EPA_disp=-30),
        )
        assert reward == -60


def test_reward_weights_from_a_real_file_shape():
    """Mirrors rewards.py: one indicator penalised at -100, the rest neutral."""
    old = indicators()
    new = indicators(v_degraded=2, v_sent_full=1)
    weights = tariffs(v_degraded=-100, v_sent_full=10)
    assert compute_reward(new, old, 1, weights) == 2 * -100 + 1 * 10
