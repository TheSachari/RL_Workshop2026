"""Tests for `Environment.fork`.

A fork has to be independent in exactly the seven containers the simulation
mutates, and share everything else -- copying the event stream or the skills
frame per actor would give back what the planning rewrite bought.

The set is `checkpoint.ENV_FIELDS`: the same containers a checkpoint saves,
for the same reason. If a field is added there it belongs here too, which
`test_fork_covers_every_checkpointed_container` pins.
"""

import pandas as pd
import pytest

from checkpoint import ENV_FIELDS
from planning_state import PlanningView, SlotStore
from sim_state import Environment


@pytest.fixture
def env():
    planning = {"S": {1: {1: {0: {"planned": [1, 2, 3],
                                  "available": [1, 2, 3],
                                  "standby": []}}}}}
    return Environment(
        dic_vehicles={"S": {"available": [10], "standby": [], "inter": []}},
        dic_functions={10: ["VSAV"]},
        df_skills=pd.DataFrame({"a": [1]}),
        dic_roles_skills={},
        dic_roles={},
        planning=PlanningView(SlotStore.from_planning(planning)),
        dic_inter={1: {}},
        dic_ff={1: 0, 2: 0, 3: 0},
        dic_indic={"v_sent": 0},
        dic_indic_old={"v_sent": 0},
        Z_1=["S"],
        Z_4=[],
        dic_lent={"S": {}},
        dic_station_distance={},
        df_pc=pd.DataFrame({"num_inter": [1]}),
        old_date=None,
        date_reference=None,
        skills_updated=None,
    )


class TestIndependence:
    def test_indicators(self, env):
        fork = env.fork()
        fork.dic_indic["v_sent"] += 1
        assert env.dic_indic["v_sent"] == 0

    def test_previous_indicators(self, env):
        fork = env.fork()
        fork.dic_indic_old["v_sent"] += 1
        assert env.dic_indic_old["v_sent"] == 0

    def test_firefighter_availability(self, env):
        fork = env.fork()
        fork.dic_ff[1] = -1
        assert env.dic_ff[1] == 0

    def test_vehicles(self, env):
        """Nested: the per-station lists must be copied, not just the outer dict."""
        fork = env.fork()
        fork.dic_vehicles["S"]["available"].append(99)
        assert env.dic_vehicles["S"]["available"] == [10]

    def test_interventions(self, env):
        fork = env.fork()
        fork.dic_inter[1]["S"] = {}
        assert env.dic_inter[1] == {}

    def test_lent(self, env):
        fork = env.fork()
        fork.dic_lent["S"][10] = [1]
        assert env.dic_lent["S"] == {}

    def test_planning(self, env):
        fork = env.fork()
        fork.planning["S"][1][1][0]["available"].remove(1)
        assert list(env.planning["S"][1][1][0]["available"]) == [1, 2, 3]

    def test_the_original_does_not_leak_into_the_fork(self, env):
        fork = env.fork()
        env.planning["S"][1][1][0]["available"].append(4)
        env.dic_ff[1] = -1
        assert list(fork.planning["S"][1][1][0]["available"]) == [1, 2, 3]
        assert fork.dic_ff[1] == 0

    def test_two_forks_are_independent_of_each_other(self, env):
        a, b = env.fork(), env.fork()
        a.dic_indic["v_sent"] = 1
        b.dic_indic["v_sent"] = 2
        assert (a.dic_indic["v_sent"], b.dic_indic["v_sent"],
                env.dic_indic["v_sent"]) == (1, 2, 0)


class TestSharing:
    """Read-only tables must be shared: copying them per actor would undo the
    saving the planning rewrite bought."""

    @pytest.mark.parametrize("field", [
        "df_pc", "df_skills", "dic_functions", "dic_roles",
        "dic_roles_skills", "dic_station_distance", "Z_1", "Z_4",
    ])
    def test_read_only_tables_are_shared(self, env, field):
        assert getattr(env.fork(), field) is getattr(env, field)


def test_fork_covers_every_checkpointed_container(env):
    """The containers a checkpoint saves are exactly the ones a fork copies.

    Both answer "what does a run mutate". A field added to one and not the
    other means either a fork sharing mutable state or a checkpoint losing it.
    """
    fork = env.fork()
    shared = [f for f in ENV_FIELDS if getattr(fork, f) is getattr(env, f)]
    assert shared == []
