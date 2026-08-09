"""Tests for the resumable training checkpoint.

A checkpoint that silently drops a field does not fail loudly -- the run
restarts, the logs look plausible, and the damage only shows up as a training
curve that never quite recovers. Every test here pins something that was
actually broken or absent before `checkpoint.py` existed:

* the target network was never saved, so any `--load` resume trained against a
  randomly initialised target and produced meaningless TD targets;
* epsilon came back without its decay counter `d`, putting the schedule on the
  wrong rung;
* buffer entries were `namedtuple`s built inside `__init__`, which pickle cannot
  resolve, and came back as CUDA tensors into a `np.stack` that needs CPU ones;
* `dic_indic_old` was easy to forget, and the reward is the delta between it and
  `dic_indic` -- lose it and the first reward after a resume is nonsense.

The fixtures here are stand-ins with the same shape as the real agent and
environment, not the real ones: building an FQF agent needs a GPU and a 3280-wide
state, and loading the environment reads the data tree. The contract under test
is what `checkpoint.save`/`restore` carry across, which the stand-ins exercise
exactly.
"""

import random
from collections import deque, namedtuple
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import checkpoint as ckpt

INDICATORS = [
    "v_required", "v_sent", "v_degraded", "rupture_ff",
    "VSAV_disp", "FPT_disp", "EPA_disp", "skill_lvl",
]


class FakeBuffer:
    """A replay buffer with the attributes `checkpoint` copies.

    Mirrors `N_Steps_Prioritized_ReplayBuffer`: a `deque` of namedtuples, an
    n-step window, a SumTree-ish object and the running counters. The
    `Experience` type is built in `__init__` exactly as the real buffers do --
    that is what makes it unpicklable at module level, and the reason
    `_plain`/`_rebuild` exist.
    """

    def __init__(self, size=8):
        self.experience = namedtuple(
            "Experience", ["state", "action", "reward", "next_state", "done"]
        )
        self.memory = deque(maxlen=size)
        self.n_step_buffer = deque(maxlen=4)
        self.sum_tree = SimpleNamespace(
            buffer_size=size, tree=np.arange(2 * size, dtype=np.float64)
        )
        self.pos = 3
        self.frame = 11
        self.beta = 0.42
        self.current_size = 0

    def add(self, i, device="cpu"):
        e = self.experience(
            torch.full((4,), float(i), device=device), i, float(i),
            torch.full((4,), float(i + 1), device=device), False,
        )
        self.memory.append(e)
        self.n_step_buffer.append(e)
        self.current_size = len(self.memory)


class FakeAgent:
    """Agent stand-in carrying every piece `_agent_state` looks for."""

    def __init__(self, buffer=None, with_extras=True):
        self.qnetwork_local = torch.nn.Linear(4, 3)
        self.qnetwork_target = torch.nn.Linear(4, 3)
        self.optimizer = torch.optim.AdamW(self.qnetwork_local.parameters(), lr=1e-3)
        self.t_step = 0
        self.memory = buffer if buffer is not None else FakeBuffer()
        if with_extras:
            # FQF-only pieces, plus the curiosity module.
            self.fpn = torch.nn.Linear(3, 2)
            self.frac_optimizer = torch.optim.RMSprop(self.fpn.parameters(), lr=1e-4)
            self.icm = torch.nn.Linear(2, 2)
            self.icm.optimizer = torch.optim.Adam(self.icm.parameters(), lr=1e-4)
        else:
            self.fpn = None
            self.frac_optimizer = None
            self.icm = None


def make_env():
    return SimpleNamespace(
        dic_vehicles={"VSAV1": "STATION_A"},
        dic_inter={7: ["ff_1", "ff_2"]},
        dic_ff={"ff_1": {"station": "A", "busy": True}, "ff_2": {"station": "B"}},
        dic_indic={k: i for i, k in enumerate(INDICATORS)},
        dic_indic_old={k: 0 for k in INDICATORS},
        planning={"A": [1, 2, 3]},
        dic_lent={"VSAV2": "STATION_B"},
        skills_updated=np.ones((2, 3)),
        old_date="2018-02-02 02:12:00",
        date_reference="2018-02-02 02:15:00",
    )


def make_args(**overrides):
    args = SimpleNamespace(
        dataset="df_pc_fake_10y_rs.pkl", start=1, end=637332, agent_model="fqf",
        n_hours=2, top_n=5, constraint_factor_veh=1, constraint_factor_ff=1, seed=41,
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def make_loop():
    return SimpleNamespace(
        vehicle_out=12, action_num=25405, current_ff_inter=["ff_1"],
        dic_log={1: "a"}, dic_back={2: "b"}, dic_start_time={3: "c"},
        dic_veh_typ={4: "d"},
    )


def save_to(tmp_path, *, agent=None, env=None, args=None, include_buffer=True, **kw):
    agent = agent if agent is not None else FakeAgent()
    env = env if env is not None else make_env()
    params = dict(
        agent=agent, env=env, fleet={"VSAV": 5}, loop=make_loop(),
        rl={"eps": 0.37, "d": 9, "score": -94100.0, "action_num": 25405},
        num_inter=5700, row_index=11405,
        old_date=env.old_date, date_reference=env.date_reference,
        reward_evo=[[1, -2.0]], dic_saved_skills={3: 7},
        args=args if args is not None else make_args(),
        include_buffer=include_buffer,
    )
    params.update(kw)
    return ckpt.save(tmp_path / "agent.ckpt", **params)


class TestRoundTrip:
    def test_position_survives(self, tmp_path):
        payload = ckpt.load(save_to(tmp_path))
        assert payload["num_inter"] == 5700
        # The row index, not the intervention counter, is what resumes the
        # stream: `num_inter` repeats across a departure/RETURN pair.
        assert payload["row_index"] == 11405

    def test_epsilon_and_its_decay_counter_travel_together(self, tmp_path):
        rl = {}
        restored = ckpt.restore(
            ckpt.load(save_to(tmp_path)), agent=FakeAgent(), env=make_env(), rl=rl
        )
        assert rl["eps"] == 0.37
        # Restoring eps without `d` silently restarts the decay schedule.
        assert rl["d"] == 9
        assert restored["num_inter"] == 5700

    def test_operational_state_survives(self, tmp_path):
        env = make_env()
        env.dic_ff["ff_1"]["busy"] = True
        fresh = make_env()
        fresh.dic_ff = {}
        payload = ckpt.load(save_to(tmp_path, env=env))
        ckpt.restore(payload, agent=FakeAgent(), env=fresh, rl={})
        assert fresh.dic_ff["ff_1"] == {"station": "A", "busy": True}
        assert fresh.dic_inter == {7: ["ff_1", "ff_2"]}
        assert fresh.dic_vehicles == {"VSAV1": "STATION_A"}
        assert fresh.planning == {"A": [1, 2, 3]}
        assert fresh.dic_lent == {"VSAV2": "STATION_B"}

    def test_indicator_pair_survives(self, tmp_path):
        """`compute_reward` reads the delta, so both halves must come back."""
        fresh = make_env()
        fresh.dic_indic, fresh.dic_indic_old = {}, {}
        ckpt.restore(ckpt.load(save_to(tmp_path)), agent=FakeAgent(), env=fresh, rl={})
        assert fresh.dic_indic == {k: i for i, k in enumerate(INDICATORS)}
        assert fresh.dic_indic_old == {k: 0 for k in INDICATORS}

    def test_dates_survive(self, tmp_path):
        fresh = make_env()
        fresh.old_date = fresh.date_reference = None
        ckpt.restore(ckpt.load(save_to(tmp_path)), agent=FakeAgent(), env=fresh, rl={})
        assert fresh.old_date == "2018-02-02 02:12:00"
        assert fresh.date_reference == "2018-02-02 02:15:00"

    def test_loop_bookkeeping_survives(self, tmp_path):
        restored = ckpt.restore(
            ckpt.load(save_to(tmp_path)), agent=FakeAgent(), env=make_env(), rl={}
        )
        assert restored["loop"]["vehicle_out"] == 12
        assert restored["loop"]["current_ff_inter"] == ["ff_1"]
        assert restored["fleet"] == {"VSAV": 5}
        assert restored["reward_evo"] == [[1, -2.0]]
        assert restored["dic_saved_skills"] == {3: 7}


class TestAgentState:
    def test_target_network_is_saved_and_restored(self, tmp_path):
        """The omission that made every previous `--load` resume unsound."""
        agent = FakeAgent()
        with torch.no_grad():
            agent.qnetwork_target.weight.fill_(0.5)
        path = save_to(tmp_path, agent=agent)

        fresh = FakeAgent()
        with torch.no_grad():
            fresh.qnetwork_target.weight.fill_(-1.0)
        ckpt.restore(ckpt.load(path), agent=fresh, env=make_env(), rl={})
        assert torch.allclose(fresh.qnetwork_target.weight, torch.full((3, 4), 0.5))

    def test_local_network_and_t_step_are_restored(self, tmp_path):
        agent = FakeAgent()
        agent.t_step = 25404
        with torch.no_grad():
            agent.qnetwork_local.weight.fill_(0.25)
        path = save_to(tmp_path, agent=agent)

        fresh = FakeAgent()
        ckpt.restore(ckpt.load(path), agent=fresh, env=make_env(), rl={})
        # t_step drives the LR schedule and the update_every cadence.
        assert fresh.t_step == 25404
        assert torch.allclose(fresh.qnetwork_local.weight, torch.full((3, 4), 0.25))

    def test_optimizer_moments_are_restored(self, tmp_path):
        agent = FakeAgent()
        agent.qnetwork_local(torch.ones(2, 4)).sum().backward()
        agent.optimizer.step()
        path = save_to(tmp_path, agent=agent)

        fresh = FakeAgent()
        assert fresh.optimizer.state_dict()["state"] == {}
        ckpt.restore(ckpt.load(path), agent=fresh, env=make_env(), rl={})
        assert fresh.optimizer.state_dict()["state"] != {}

    def test_fqf_and_curiosity_pieces_are_restored(self, tmp_path):
        agent = FakeAgent()
        with torch.no_grad():
            agent.fpn.weight.fill_(0.75)
            agent.icm.weight.fill_(0.125)
        path = save_to(tmp_path, agent=agent)

        fresh = FakeAgent()
        ckpt.restore(ckpt.load(path), agent=fresh, env=make_env(), rl={})
        assert torch.allclose(fresh.fpn.weight, torch.full((2, 3), 0.75))
        assert torch.allclose(fresh.icm.weight, torch.full((2, 2), 0.125))

    def test_agent_without_optional_pieces_round_trips(self, tmp_path):
        """DQN/PPO have no fpn or ICM; their absence must not break saving."""
        path = save_to(tmp_path, agent=FakeAgent(with_extras=False))
        fresh = FakeAgent(with_extras=False)
        ckpt.restore(ckpt.load(path), agent=fresh, env=make_env(), rl={})
        assert fresh.t_step == 0


class TestReplayBuffer:
    def test_transitions_and_counters_survive(self, tmp_path):
        agent = FakeAgent()
        for i in range(5):
            agent.memory.add(i)
        path = save_to(tmp_path, agent=agent)

        fresh = FakeAgent()
        ckpt.restore(ckpt.load(path), agent=fresh, env=make_env(), rl={})
        assert len(fresh.memory.memory) == 5
        # The partial n-step window matters: a transition only enters memory
        # once the window closes.
        assert len(fresh.memory.n_step_buffer) == 4
        assert fresh.memory.pos == 3
        assert fresh.memory.frame == 11
        assert fresh.memory.beta == 0.42
        expected_tree = np.arange(16, dtype=np.float64)
        assert np.array_equal(fresh.memory.sum_tree.tree, expected_tree)

    def test_entries_come_back_as_the_live_experience_type(self, tmp_path):
        """Stored as plain tuples (pickle cannot see the namedtuple class),
        rebuilt against the live buffer so `sample()` can use field access."""
        agent = FakeAgent()
        agent.memory.add(1)
        path = save_to(tmp_path, agent=agent)

        fresh = FakeAgent()
        ckpt.restore(ckpt.load(path), agent=fresh, env=make_env(), rl={})
        entry = fresh.memory.memory[0]
        assert isinstance(entry, tuple)
        assert entry.action == 1
        assert torch.allclose(entry.state, torch.ones(4))

    def test_stored_states_are_cpu_tensors(self, tmp_path):
        """`sample()` calls `np.stack` on the states, which fails on CUDA."""
        agent = FakeAgent()
        agent.memory.add(1)
        path = save_to(tmp_path, agent=agent)

        fresh = FakeAgent()
        ckpt.restore(ckpt.load(path), agent=fresh, env=make_env(), rl={})
        states = [e.state for e in fresh.memory.memory]
        assert all(s.device.type == "cpu" for s in states)
        np.stack(states)  # would raise if any tensor were on the GPU

    def test_buffer_can_be_omitted(self, tmp_path):
        agent = FakeAgent()
        agent.memory.add(1)
        payload = ckpt.load(save_to(tmp_path, agent=agent, include_buffer=False))
        assert "memory" not in payload["agent"]

        # Restoring must leave the live buffer alone rather than clearing it.
        fresh = FakeAgent()
        fresh.memory.add(9)
        ckpt.restore(payload, agent=fresh, env=make_env(), rl={})
        assert len(fresh.memory.memory) == 1


class TestRngStreams:
    def test_rng_states_are_restored(self, tmp_path):
        """`act()` draws from np.random/random; `sample()` from the buffer's."""
        random.seed(1234)
        np.random.seed(1234)
        torch.manual_seed(1234)
        path = save_to(tmp_path)

        expected = (random.random(), float(np.random.rand()), float(torch.rand(1)))

        # Advance all three streams well past the checkpoint.
        for _ in range(50):
            random.random(), np.random.rand(), torch.rand(1)

        ckpt.restore(ckpt.load(path), agent=FakeAgent(), env=make_env(), rl={})
        got = (random.random(), float(np.random.rand()), float(torch.rand(1)))
        assert got == pytest.approx(expected)


class TestCompatibility:
    @pytest.mark.parametrize(
        "field,value",
        [
            ("dataset", "df_pc_fake_test_rs.pkl"),
            ("end", 127000),  # rescales eps_update and max_train_steps
            ("seed", 7),
            ("constraint_factor_ff", 3),
            ("agent_model", "dqn"),
        ],
    )
    def test_mismatched_settings_are_refused(self, tmp_path, field, value):
        payload = ckpt.load(save_to(tmp_path))
        with pytest.raises(ValueError, match="different settings"):
            ckpt.check_compatible(payload, make_args(**{field: value}))

    def test_matching_settings_pass(self, tmp_path):
        ckpt.check_compatible(ckpt.load(save_to(tmp_path)), make_args())

    def test_version_mismatch_is_refused(self, tmp_path):
        path = save_to(tmp_path)
        payload = torch.load(path, weights_only=False)
        payload["version"] = ckpt.CKPT_VERSION + 1
        torch.save(payload, path)
        with pytest.raises(ValueError, match="version"):
            ckpt.load(path)


class TestDurability:
    def test_each_save_replaces_the_last(self, tmp_path):
        """One checkpoint on disk, by default: saves overwrite each other."""
        first = save_to(tmp_path, num_inter=10000)
        second = save_to(tmp_path, num_inter=20000)
        assert first == second  # same live path
        assert ckpt.load(second)["num_inter"] == 20000
        assert not (tmp_path / "agent.ckpt.prev").exists()
        assert [p.name for p in tmp_path.iterdir()] == ["agent.ckpt"]

    def test_previous_checkpoint_is_kept_when_asked(self, tmp_path):
        save_to(tmp_path, num_inter=10000, keep=2)
        second = save_to(tmp_path, num_inter=20000, keep=2)
        assert ckpt.load(second)["num_inter"] == 20000
        assert ckpt.load(tmp_path / "agent.ckpt.prev")["num_inter"] == 10000

    def test_dropping_to_one_removes_a_stale_prev(self, tmp_path):
        save_to(tmp_path, num_inter=10000, keep=2)
        save_to(tmp_path, num_inter=20000, keep=2)
        assert (tmp_path / "agent.ckpt.prev").exists()
        save_to(tmp_path, num_inter=30000)  # back to the default
        assert not (tmp_path / "agent.ckpt.prev").exists()

    def test_no_temp_file_is_left_behind(self, tmp_path):
        save_to(tmp_path)
        assert not list(tmp_path.glob("*.tmp"))
