"""Tests for the decision-log analysis.

Most of this module prints, and printing is not worth pinning. What is worth
pinning is the one place it makes a judgement: whether two runs are comparable
at all. Cumulative counters make a shorter run look uniformly "better", so a
verdict printed against a mismatched baseline would be confidently wrong -- the
kind of error that survives into a paper because the number looks plausible.
"""

import pytest

from analyse_decisions import LOWER_IS_BETTER, comparable, pct


class TestComparability:
    def test_equal_workloads_are_comparable(self):
        ok, why = comparable({"v_required": 800}, {"v_required": 800})
        assert ok and why == ""

    @pytest.mark.parametrize("other", [840, 760])
    def test_small_differences_are_tolerated(self, other):
        """Two runs over the same window still differ a little."""
        ok, _ = comparable({"v_required": 800}, {"v_required": other})
        assert ok

    @pytest.mark.parametrize("other", [86032, 100])
    def test_different_workloads_are_refused(self, other):
        ok, why = comparable({"v_required": 805}, {"v_required": other})
        assert not ok
        assert "v_required" in why

    def test_a_missing_workload_does_not_block(self):
        """Nothing to check against: report, rather than refuse silently."""
        ok, _ = comparable({}, {"v_required": 800})
        assert ok
        ok, _ = comparable({"v_required": 0}, {"v_required": 800})
        assert ok


class TestVerdictScope:
    def test_only_cost_indicators_carry_a_verdict(self):
        assert "rupture_ff" in LOWER_IS_BETTER
        assert "v_degraded" in LOWER_IS_BETTER

    def test_volume_indicators_do_not(self):
        """Sending more crews is not by itself better or worse."""
        for k in ("v_sent", "ff_sent", "skill_lvl", "v_required"):
            assert k not in LOWER_IS_BETTER


def test_pct_handles_an_empty_run():
    assert pct(0, 0) == "n/a"
    assert pct(1, 4) == "25.0%"
