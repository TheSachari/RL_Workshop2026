"""Tests for the PPO surrogate objective.

The agent that lived here before was named PPO but was REINFORCE with a
baseline: no importance ratio, no clipping, one gradient pass per rollout, and
Monte-Carlo returns truncated at the batch boundary. Each of those silently
degrades rather than fails -- the run trains, the curve moves, and only the
sample efficiency and the stability are worse than the name promises. Every
test here pins one of the pieces that makes it actually PPO:

* the behaviour log-prob must be recorded at acting time, because recomputing
  it after the update gives a ratio of exactly 1 and disables clipping;
* the rollout must be reused for several epochs, or the clip has nothing to
  constrain;
* the clip must bind when the policy moves far, which is the whole point on a
  reward whose `rupture_ff` term is -100 and sparse;
* the rollout is cut every `batch_size` decisions, not at an intervention end,
  so a non-terminal cut must bootstrap its value instead of assuming zero.

`get_potential_actions` is stubbed: it reads the simulator's state encoding,
and what these tests need from it is only a feasible-action mask.
"""

import numpy as np
import pytest
import torch

import agent_explainable as AE
from agent_explainable import PPOAgent

D_INPUT = 3280 // 82


def make_agent(**overrides):
    kwargs = dict(
        state_size=3280, action_size=80, layer_type="ff", layer_size=32,
        num_layers=2, use_batchnorm=True, am=True, n_steps=4, batch_size=32,
        buffer_size=100, lr=1e-3, lr_dec=1, tau=0.005, gamma=0.99,
        munchausen=0, curiosity=0, curiosity_size=32, per=2, rdm=0,
        entropy_tau=0.03, entropy_tau_coeff=0.01, lo=-1, alpha=0.9,
        n_quantiles=32, entropy_coeff=0.01, update_every=128,
        max_train_steps=1000, decay_update=100, device=torch.device("cpu"),
        seed=41, clip_range=0.2, n_epochs=4, minibatch_size=8,
        gae_lambda=0.95, value_coeff=0.5, target_kl=None,
    )
    kwargs.update(overrides)
    return PPOAgent(**kwargs)


def make_state(rng):
    state = np.zeros((82, D_INPUT), dtype=np.float32)
    state[2:, 0] = rng.random(80)
    return state


@pytest.fixture
def feasible(monkeypatch):
    """Stub the mask to a fixed feasible set."""
    actions = list(range(8))
    monkeypatch.setattr(
        AE, "get_potential_actions", lambda state, waiting: (actions, [1] * len(actions))
    )
    return actions


class TestBehaviourPolicy:
    def test_act_records_log_prob_and_value(self, feasible):
        agent = make_agent()
        rng = np.random.default_rng(0)
        action, _skill, potential = agent.act(make_state(rng), False)

        assert action in potential
        # A log-prob is negative; 0.0 would claim certainty and flatten the ratio.
        assert agent.last_log_prob is not None and agent.last_log_prob < 0.0
        assert agent.last_value is not None

    def test_step_without_act_is_dropped(self, feasible):
        """A transition with no behaviour log-prob cannot form a ratio."""
        agent = make_agent()
        rng = np.random.default_rng(0)
        agent.step(make_state(rng), 0, 1.0, make_state(rng), False)
        assert agent.rollout_storage == []

    def test_sampled_action_is_always_feasible(self, monkeypatch):
        """Masked actions must never be sampled: they map to no firefighter."""
        monkeypatch.setattr(
            AE, "get_potential_actions", lambda s, w: ([3, 17, 42], [1, 1, 1])
        )
        agent = make_agent()
        rng = np.random.default_rng(1)
        for _ in range(30):
            action, _skill, _pot = agent.act(make_state(rng), False)
            assert action in (3, 17, 42)


class TestClippedSurrogate:
    def test_rollout_is_reused_across_epochs(self, feasible):
        """One pass per rollout throws away data the simulator paid for."""
        agent = make_agent(n_epochs=5, batch_size=32, minibatch_size=8)
        seen = []
        original = agent.model.forward
        agent.model.forward = lambda s: (seen.append(s.shape[0]), original(s))[1]

        rng = np.random.default_rng(2)
        for _ in range(32):
            state, next_state = make_state(rng), make_state(rng)
            action, _skill, _pot = agent.act(state, False)
            agent.step(state, action, float(rng.normal()), next_state, False)

        # 5 epochs x 4 minibatches x (actor + critic) forward passes.
        minibatch_passes = [n for n in seen if n == 8]
        assert len(minibatch_passes) == 5 * 4 * 2

    def test_clip_binds_when_policy_moves(self, feasible):
        """With a strongly polarised reward the ratio must leave the band."""
        agent = make_agent(n_epochs=8, batch_size=64, minibatch_size=16,
                           lr=1e-2, entropy_coeff=0.0)
        ratios = []
        original = torch.clamp

        def spy(tensor, *args, **kwargs):
            if args and args[0] == pytest.approx(0.8):
                ratios.append(tensor.detach().clone())
            return original(tensor, *args, **kwargs)

        rng = np.random.default_rng(3)
        torch.clamp = spy
        try:
            for _ in range(64):
                state, next_state = make_state(rng), make_state(rng)
                action, _skill, _pot = agent.act(state, False)
                agent.step(state, action, 10.0 if action == 0 else -10.0,
                           next_state, False)
        finally:
            torch.clamp = original

        assert ratios, "clamp never reached: the surrogate is not clipped"
        observed = torch.cat([r.reshape(-1) for r in ratios])
        assert (observed - 1.0).abs().max() > 1e-6, "ratios pinned at 1.0"
        outside = ((observed < 0.8) | (observed > 1.2)).float().mean()
        assert outside > 0.0, "clip never bound"

    def test_update_returns_finite_losses_and_moves_policy(self, feasible):
        agent = make_agent()
        before = [p.clone() for p in agent.model.policy_head.parameters()]

        rng = np.random.default_rng(4)
        losses = []
        for step in range(64):
            state, next_state = make_state(rng), make_state(rng)
            action, _skill, _pot = agent.act(state, False)
            out = agent.step(state, action, float(rng.normal()), next_state,
                             bool(step % 17 == 0))
            if out is not None:
                losses.append(out)

        assert len(losses) == 2
        for actor_loss, critic_loss in losses:
            assert torch.isfinite(actor_loss) and torch.isfinite(critic_loss)
        after = list(agent.model.policy_head.parameters())
        assert any(not torch.allclose(b, a) for b, a in zip(before, after))

    def test_rollout_is_cleared_after_update(self, feasible):
        """Reusing a rollout past its update would make the data off-policy."""
        agent = make_agent(batch_size=16)
        rng = np.random.default_rng(5)
        for _ in range(16):
            state, next_state = make_state(rng), make_state(rng)
            action, _skill, _pot = agent.act(state, False)
            agent.step(state, action, 1.0, next_state, False)
        assert agent.rollout_storage == []


class TestAdvantages:
    def test_terminal_state_is_not_bootstrapped(self, feasible):
        agent = make_agent()
        assert agent._bootstrap_value(torch.zeros(82, D_INPUT), True) == 0.0

    def test_non_terminal_cut_is_bootstrapped(self, feasible):
        """The rollout boundary is arbitrary: truncating there biases the return."""
        agent = make_agent()
        value = agent._bootstrap_value(torch.zeros(82, D_INPUT), False)
        assert value != 0.0

    def test_early_stop_on_kl(self, feasible):
        """A large policy drift must cut the epoch loop short."""
        agent = make_agent(n_epochs=20, batch_size=32, minibatch_size=8,
                           lr=1e-1, entropy_coeff=0.0, target_kl=1e-6)
        passes = []
        original = agent.model.forward
        agent.model.forward = lambda s: (passes.append(s.shape[0]), original(s))[1]

        rng = np.random.default_rng(6)
        for _ in range(32):
            state, next_state = make_state(rng), make_state(rng)
            action, _skill, _pot = agent.act(state, False)
            agent.step(state, action, 10.0 if action == 0 else -10.0,
                       next_state, False)

        assert len([n for n in passes if n == 8]) < 20 * 4 * 2


class TestMunchausen:
    def test_munchausen_is_refused(self, feasible):
        """Munchausen rewrites the reward for a soft-greedy value target; under
        PPO that biases the advantage the surrogate is built on."""
        agent = make_agent(munchausen=1, batch_size=4)
        rng = np.random.default_rng(7)
        with pytest.raises(ValueError, match="munchausen"):
            for _ in range(4):
                state, next_state = make_state(rng), make_state(rng)
                action, _skill, _pot = agent.act(state, False)
                agent.step(state, action, 1.0, next_state, False)


class TestCheckpoint:
    """PPO has no replay buffer and two optimisers instead of one.

    `checkpoint.save` reached for `agent.memory` unconditionally, so a PPO run
    died with `AttributeError` at its first `--checkpoint_every` boundary --
    an hour into training rather than at startup.
    """

    def test_ppo_agent_checkpoints_without_a_replay_buffer(self, feasible, tmp_path):
        import checkpoint as ckpt

        agent = make_agent(batch_size=8)
        rng = np.random.default_rng(8)
        for _ in range(8):
            state, next_state = make_state(rng), make_state(rng)
            action, _skill, _pot = agent.act(state, False)
            agent.step(state, action, 1.0, next_state, False)

        state_dict = ckpt._agent_state(agent, include_buffer=True)
        assert "memory" not in state_dict
        # Both optimisers must ride along, or Adam's moments restart at zero.
        assert "actor_optimizer" in state_dict
        assert "critic_optimizer" in state_dict

        fresh = make_agent(batch_size=8)
        ckpt._restore_agent(fresh, state_dict)
        assert fresh.t_step == agent.t_step
        for before, after in zip(
            agent.model.policy_head.parameters(), fresh.model.policy_head.parameters()
        ):
            assert torch.allclose(before, after)


class TestSharedEncoder:
    """Actor and critic have disjoint bodies and heads, but share the attention
    block and the two row encoders that feed them.

    Listing only the bodies and heads in the optimisers left those parameters --
    the attention that compares the 80 candidates against each other -- in no
    optimiser at all. They stayed at their random initialisation for a whole
    training run, and the failure was silent: the loss moved, the run completed,
    and only the shortfall curve drifting the wrong way gave it away.
    """

    def test_every_parameter_is_owned_by_exactly_one_optimiser(self, feasible):
        agent = make_agent()
        actor_ids = {id(p) for p in agent.actor_parameters}
        critic_ids = {id(p) for p in agent.critic_parameters}

        # In both would take two Adam steps per minibatch off inconsistent moments.
        assert not (actor_ids & critic_ids)
        every = {id(p) for _, p in agent.model.named_parameters()}
        assert not (every - actor_ids - critic_ids)

    def test_shared_encoder_trains(self, feasible):
        agent = make_agent(batch_size=32, minibatch_size=8)
        shared = "attention.multihead_attn.in_proj_weight"
        before = dict(agent.model.named_parameters())[shared].detach().clone()

        rng = np.random.default_rng(9)
        for _ in range(32):
            # Header rows non-zero: a zero row gives its encoder no gradient,
            # which would hide a genuinely detached parameter.
            state = rng.random((82, D_INPUT)).astype(np.float32)
            next_state = rng.random((82, D_INPUT)).astype(np.float32)
            action, _skill, _pot = agent.act(state, False)
            agent.step(state, action, float(rng.normal()), next_state, False)

        after = dict(agent.model.named_parameters())[shared]
        assert not torch.allclose(before, after), "shared attention never updated"
