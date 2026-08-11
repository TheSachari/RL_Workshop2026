"""Tests for what the n-step prioritized buffer actually stores.

Two properties, both invisible from the training curves and both easy to break:
the reward kept must be the n-step return the buffer computes, and the states
must be stored compressed without that leaking into the learning path.
"""

import numpy as np
import torch

from ReplayBuffers import N_Steps_Prioritized_ReplayBuffer

GAMMA = 0.9
N_STEPS = 3
STATE = 8


def buffer(**kw):
    params = dict(buffer_size=50, batch_size=4, seed=0, gamma=GAMMA, n_steps=N_STEPS)
    params.update(kw)
    return N_Steps_Prioritized_ReplayBuffer(**params)


def fill(buf, rewards):
    for i, r in enumerate(rewards):
        buf.add(torch.full((STATE,), float(i)), i % 3, r, torch.rand(STATE), False)


class TestNStepReturn:
    """The stored reward is the discounted n-step return, not the last step's.

    `calc_multistep_return` returns it as `n_steps_reward`, but the call
    unpacked it and then built the experience from `reward` -- the caller's loop
    variable, still holding the most recent single-step reward. The n-step
    return was computed and dropped, leaving `n_steps` to affect only the
    discount exponent in `learn_per` and never the target it is compared to.
    """

    def test_the_stored_reward_is_the_discounted_return(self):
        buf = buffer()
        fill(buf, [1.0, 2.0, 4.0])
        expected = 1.0 + GAMMA * 2.0 + GAMMA**2 * 4.0
        assert buf.memory[0].reward == expected

    def test_it_is_not_the_last_single_step_reward(self):
        """The regression, stated as its own case: 4.0 would mean the bug is back."""
        buf = buffer()
        fill(buf, [1.0, 2.0, 4.0])
        assert buf.memory[0].reward != 4.0

    def test_each_entry_starts_from_its_own_window(self):
        buf = buffer()
        fill(buf, [1.0, 2.0, 4.0, 8.0])
        assert buf.memory[1].reward == 2.0 + GAMMA * 4.0 + GAMMA**2 * 8.0

    def test_nothing_is_stored_before_the_window_is_full(self):
        buf = buffer()
        fill(buf, [1.0, 2.0])
        assert len(buf) == 0


class TestStateCompression:
    """States are kept as float16; learning still receives float32."""

    def test_states_are_stored_as_float16(self):
        buf = buffer()
        fill(buf, [1.0, 2.0, 4.0])
        entry = buf.memory[0]
        assert entry.state.dtype == np.float16
        assert entry.next_state.dtype == np.float16

    def test_the_values_survive_the_round_trip(self):
        """fp16 carries ~3 decimal digits, ample for normalised features."""
        buf = buffer()
        state = torch.tensor([0.0, 1.0, 0.5, 0.125, -1.0, 0.333, 0.75, 0.001])
        for _ in range(N_STEPS):
            buf.add(state, 0, 1.0, state, False)
        stored = torch.as_tensor(buf.memory[0].state, dtype=torch.float32)
        assert torch.allclose(stored, state, atol=1e-3)

    def test_a_sampled_batch_converts_to_float32_cleanly(self):
        buf = buffer()
        fill(buf, [float(i) for i in range(12)])
        states, _, _, _, _, _, _ = buf.sample()
        batch = torch.as_tensor(np.stack(states), dtype=torch.float32)
        assert batch.dtype == torch.float32
        assert torch.isfinite(batch).all()

    def test_rewards_keep_full_precision(self):
        """Only states are compressed: a 64-term discounted sum would round."""
        buf = buffer()
        fill(buf, [0.1, 0.2, 0.3])
        assert isinstance(buf.memory[0].reward, float)
        expected = 0.1 + GAMMA * 0.2 + GAMMA**2 * 0.3
        assert abs(buf.memory[0].reward - expected) < 1e-12

    def test_numpy_states_are_accepted_too(self):
        """`add` is called with tensors today; arrays must not silently break."""
        buf = buffer()
        for i in range(N_STEPS):
            buf.add(np.full(STATE, float(i), dtype=np.float32), 0, 1.0,
                    np.zeros(STATE, dtype=np.float32), False)
        assert buf.memory[0].state.dtype == np.float16
