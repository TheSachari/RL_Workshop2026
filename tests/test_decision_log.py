"""Tests for the per-decision explanation log.

The log's value rests on two claims: that it records the quantities the agent
actually decided from, and that its sampling keeps the informative decisions.
Both are easy to break silently -- a wrong action-to-firefighter mapping names
the wrong person in an explanation, and a sampling rule that quietly drops
skill-relevant cases leaves a log that looks fine and misses the point.
"""

import numpy as np
import pytest

from decision_log import DecisionLog, describe


def base_call(**overrides):
    """A decision with two feasible firefighters and no rare skill in play."""
    call = dict(
        num_inter=412331,
        idx_role=2,
        num_d=1,
        current_station="TOULOUSE - VION",
        v_mat=2004,
        date="2018-02-02 02:12:00",
        action=0,
        potential_actions=[0, 1],
        ff_existing=[2871, 1096],
        dic_rare_skills={0: [], 1: []},
        upcoming_skills=np.array([], dtype=int),
        skill_lvl=3.0,
        explain={"random": False, "q_values": {0: 3.4, 1: 3.1}, "quantiles": None},
    )
    call.update(overrides)
    return call


def log(tmp_path, **kw):
    return DecisionLog(tmp_path / "decisions.pkl", **kw)


class TestWhatIsRecorded:
    def test_forced_decisions_are_counted_but_not_recorded(self, tmp_path):
        """One feasible firefighter is not a choice, so there is nothing to explain."""
        dl = log(tmp_path, rate=1.0)
        dl.consider(**base_call(potential_actions=[0]))
        assert dl.records == []
        assert dl.summary()["forced"] == 1
        assert dl.summary()["decisions_seen"] == 1

    def test_no_assignable_firefighter_is_not_recorded(self, tmp_path):
        dl = log(tmp_path, rate=1.0)
        dl.consider(**base_call(action=79, potential_actions=[79]))
        assert dl.records == []
        assert dl.summary()["forced"] == 1

    def test_action_maps_to_the_firefighter_it_designates(self, tmp_path):
        """An action is a position in `ff_existing`, not an identifier."""
        dl = log(tmp_path, rate=1.0)
        dl.consider(**base_call(action=1))
        rec = dl.records[0]
        assert rec["ff_chosen"] == 1096
        assert rec["ff_feasible"] == {0: 2871, 1: 1096}

    def test_margin_is_the_gap_to_the_best_alternative(self, tmp_path):
        dl = log(tmp_path, rate=1.0)
        dl.consider(**base_call())
        assert dl.records[0]["margin"] == pytest.approx(0.3)

    def test_exploratory_actions_carry_no_q_values(self, tmp_path):
        """An epsilon-greedy pick has no Q behind it; the log must not imply one."""
        dl = log(tmp_path, rate=1.0)
        dl.consider(**base_call(
            explain={"random": True, "q_values": None, "quantiles": None}
        ))
        rec = dl.records[0]
        assert rec["exploratory"] is True
        assert rec["q_values"] is None
        assert rec["margin"] is None

    def test_quantiles_are_kept_only_when_asked(self, tmp_path):
        explain = {
            "random": False,
            "q_values": {0: 3.4, 1: 3.1},
            "quantiles": {0: np.zeros(32), 1: np.ones(32)},
        }
        off = log(tmp_path, rate=1.0)
        off.consider(**base_call(explain=explain))
        assert "quantiles" not in off.records[0]

        on = log(tmp_path, rate=1.0, keep_quantiles=True)
        on.consider(**base_call(explain=explain))
        assert on.records[0]["quantiles"][1].shape == (32,)


class TestRareSkillCounterfactual:
    def test_records_what_the_alternative_would_have_brought(self, tmp_path):
        dl = log(tmp_path, rate=0.0)
        dl.consider(**base_call(
            dic_rare_skills={0: [7], 1: [31, 73]},
            upcoming_skills=np.array([31, 73]),
        ))
        rec = dl.records[0]
        # Firefighter 1096 (action 1) carried both, and was not chosen.
        assert rec["rare_skills_given_up"] == {1: [31, 73]}
        assert rec["why"]["skill"] is True

    # margin=0 so only the rare-skill rule can keep these; the fixture's
    # Q-values are 0.3 apart and would otherwise register as a close call.
    def test_a_skill_the_chosen_one_also_has_is_not_given_up(self, tmp_path):
        dl = log(tmp_path, rate=0.0, margin=0.0)
        dl.consider(**base_call(
            dic_rare_skills={0: [31], 1: [31]},
            upcoming_skills=np.array([31]),
        ))
        assert dl.records == []  # nothing at stake, nothing sampled

    def test_a_rare_skill_the_window_does_not_need_is_ignored(self, tmp_path):
        dl = log(tmp_path, rate=0.0, margin=0.0)
        dl.consider(**base_call(
            dic_rare_skills={0: [], 1: [99]},
            upcoming_skills=np.array([31]),
        ))
        assert dl.records == []


class TestSkillsOfInterest:
    """Restricting which skills count as worth reporting.

    "Rare" is one threshold fixed when the dataset was built, and at that
    cut-off the rule matched 78% of the test year -- it does not separate a
    skill held by 79 firefighters from one held by three.
    """

    def test_a_skill_outside_the_set_does_not_trigger_the_rule(self, tmp_path):
        dl = log(tmp_path, rate=0.0, margin=0.0, skills_of_interest={90})
        dl.consider(**base_call(
            dic_rare_skills={0: [], 1: [32]},   # common skill
            upcoming_skills=np.array([32]),
        ))
        assert dl.records == []

    def test_a_skill_inside_the_set_still_triggers_it(self, tmp_path):
        dl = log(tmp_path, rate=0.0, margin=0.0, skills_of_interest={90})
        dl.consider(**base_call(
            dic_rare_skills={0: [], 1: [90]},
            upcoming_skills=np.array([90]),
        ))
        assert dl.records[0]["rare_skills_given_up"] == {1: [90]}

    def test_none_means_every_rare_skill(self, tmp_path):
        dl = log(tmp_path, rate=0.0, margin=0.0, skills_of_interest=None)
        dl.consider(**base_call(
            dic_rare_skills={0: [], 1: [32]},
            upcoming_skills=np.array([32]),
        ))
        assert len(dl.records) == 1

    def test_the_set_is_reported(self, tmp_path):
        dl = log(tmp_path, skills_of_interest={90, 59})
        assert dl.summary()["skills_of_interest"] == [59, 90]


class TestRarestSkills:
    def test_ranks_by_how_often_a_skill_is_given_up(self, tmp_path):
        dl = log(tmp_path, rate=1.0, margin=0.0)
        # skill 5 appears three times, skill 7 once.
        for skills in ([5], [5], [5], [7]):
            dl.consider(**base_call(
                dic_rare_skills={0: [], 1: skills},
                upcoming_skills=np.array(skills),
            ))
        path = dl.flush()

        from decision_log import rarest_skills
        assert rarest_skills(path, percentile=50) == {7}

    def test_an_empty_log_yields_an_empty_set(self, tmp_path):
        from decision_log import rarest_skills
        path = log(tmp_path).flush()
        assert rarest_skills(path) == set()


class TestSampling:
    def test_close_decisions_are_kept_without_sampling(self, tmp_path):
        dl = log(tmp_path, rate=0.0, margin=0.5)
        dl.consider(**base_call(
            explain={"random": False, "q_values": {0: 3.4, 1: 3.35}, "quantiles": None}
        ))
        assert len(dl.records) == 1
        assert dl.records[0]["why"]["close"] is True

    def test_clear_decisions_are_dropped_when_nothing_else_applies(self, tmp_path):
        dl = log(tmp_path, rate=0.0, margin=0.5)
        dl.consider(**base_call(
            explain={"random": False, "q_values": {0: 9.0, 1: 1.0}, "quantiles": None}
        ))
        assert dl.records == []
        assert dl.summary()["decisions_seen"] == 1

    def test_rate_one_keeps_everything_left(self, tmp_path):
        dl = log(tmp_path, rate=1.0, margin=0.0)
        for _ in range(5):
            dl.consider(**base_call())
        assert len(dl.records) == 5

    def test_sampling_is_reproducible(self, tmp_path):
        kept = []
        for _ in range(2):
            dl = log(tmp_path, rate=0.5, margin=0.0, seed=7)
            for _ in range(40):
                dl.consider(**base_call())
            kept.append(len(dl.records))
        assert kept[0] == kept[1]

    def test_cap_stops_collecting_but_keeps_counting(self, tmp_path):
        dl = log(tmp_path, rate=1.0, max_records=3)
        for _ in range(10):
            dl.consider(**base_call())
        assert len(dl.records) == 3
        assert dl.summary()["dropped_over_cap"] == 7
        assert dl.summary()["decisions_seen"] == 10


class TestOutput:
    def test_flush_writes_records_and_summary(self, tmp_path):
        import pickle

        dl = log(tmp_path, rate=1.0)
        dl.consider(**base_call())
        path = dl.flush()

        payload = pickle.loads(path.read_bytes())
        assert payload["summary"]["recorded"] == 1
        assert payload["records"][0]["ff_chosen"] == 2871
        assert not list(tmp_path.glob("*.tmp"))

    def test_flush_is_repeatable(self, tmp_path):
        """Called at every checkpoint, so it must overwrite cleanly."""
        import pickle

        dl = log(tmp_path, rate=1.0)
        dl.consider(**base_call())
        dl.flush()
        dl.consider(**base_call())
        path = dl.flush()
        assert len(pickle.loads(path.read_bytes())["records"]) == 2

    def test_describe_names_the_firefighters(self, tmp_path):
        dl = log(tmp_path, rate=1.0)
        dl.consider(**base_call(
            dic_rare_skills={0: [], 1: [31]},
            upcoming_skills=np.array([31]),
        ))
        text = describe(dl.records[0])
        assert "2871" in text          # chosen
        assert "1096" in text          # the alternative that carried the skill
        assert "31" in text

    def test_describe_flags_an_exploratory_action(self, tmp_path):
        dl = log(tmp_path, rate=1.0)
        dl.consider(**base_call(
            explain={"random": True, "q_values": None, "quantiles": None}
        ))
        assert "exploratory" in describe(dl.records[0])
