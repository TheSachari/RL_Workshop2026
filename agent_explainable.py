"""
RL agents (DQN / FQF / Decision Transformer / PPO) for the firefighter dispatch simulator.

Notes
-----
This file depends on project-local modules:
- networks.py
- ReplayBuffers.py
- IntrinsicCuriosityModule.py
- collective_functions.py

If you rename public classes, keep the backward-compatible aliases at the bottom.
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import schedulefree
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from collective_functions import get_potential_actions
from IntrinsicCuriosityModule import ICM, Forward, Inverse
from networks import (
    FPN,
    QVN,
    DT_Network,
    Dueling_QNetwork,
    PPO_ActorCriticAM,
    PPOActorCritic,
)
from ReplayBuffers import (
    DT_ReplayBuffer,
    N_Steps_Prioritized_ReplayBuffer,
    PrioritizedReplay,
    ReplayBuffer,
)


# -----------------------------
# Utility helpers
# -----------------------------
def _norm_layers(model: torch.nn.Module) -> Tuple[torch.nn.Module, ...]:
    """The modules of `model` whose behaviour actually depends on train/eval.

    Cached on the model, since the set never changes after construction.
    """
    cached = getattr(model, "_train_sensitive_modules", None)
    if cached is None:
        cached = tuple(
            m for m in model.modules()
            if isinstance(m, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.Dropout))
        )
        model._train_sensitive_modules = cached
    return cached


class inference_pass:
    """Run a forward pass in eval mode, restoring training mode afterwards.

    `model.eval()` / `model.train()` walk the whole module tree and assign
    `self.training` on every submodule. Around a single-state forward that is
    559k `train()` calls and 560k `__setattr__` per 3k interventions -- 6 s of a
    60 s training run, 10%, to flip a flag that only batch-norm and dropout
    read.

    Only those modules are toggled here, and only the ones that were actually
    training are restored, so a model deliberately left in eval stays there.
    Pairs with `torch.inference_mode()` at the call sites, which is what
    disables autograd; this class only handles the mode.
    """

    __slots__ = ("_was_training",)

    def __init__(self, model: torch.nn.Module) -> None:
        self._was_training = [m for m in _norm_layers(model) if m.training]

    def __enter__(self) -> "inference_pass":
        for m in self._was_training:
            m.training = False
        return self

    def __exit__(self, *exc) -> None:
        for m in self._was_training:
            m.training = True


def _best_feasible_action(q: Tensor, potential_actions: Sequence[int]) -> int:
    """`filter_q_values` without moving the whole Q vector off the device.

    Ties matter: `max(..., key=...)` keeps the *first* action of
    `potential_actions` among equals, and equal Q-values are common early in
    training when the network is near-untrained. `torch.argmax` returns the
    lowest *index*, which is a different action whenever `potential_actions` is
    not sorted, so the winner is chosen by comparing values in the caller's
    order rather than by argmax.
    """
    if len(potential_actions) == 1:
        return int(potential_actions[0])
    # Historical special-case kept for backward compatibility.
    if list(potential_actions) == [79]:
        return 79

    idx = torch.as_tensor(potential_actions, dtype=torch.long, device=q.device)
    values = q.flatten().index_select(0, idx)
    # One transfer of len(potential_actions) floats, not the full action space.
    best = max(range(len(potential_actions)), key=values.tolist().__getitem__)
    return int(potential_actions[best])


def _as_tensor(x: Union[np.ndarray, Tensor], device: torch.device) -> Tensor:
    """Convert numpy arrays to float32 tensors on the given device.

    If `x` is already a tensor, it is returned (moved to `device` if needed).
    """
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def filter_q_values(q_values: Sequence[float], potential_actions: Sequence[int]) -> int:
    """Select the argmax among valid actions only.

    Parameters
    ----------
    q_values:
        Full list of Q-values for all actions.
    potential_actions:
        Subset of action indices that are currently valid.

    Returns
    -------
    int
        The best action among `potential_actions`.
    """
    # Historical special-case kept for backward compatibility.
    if list(potential_actions) == [79]:
        return 79

    best_action = max(potential_actions, key=lambda a: q_values[a])
    return int(best_action)


class RunningMeanStd:
    """Welford accumulator for a scalar stream.

    Used to normalise PPO's returns by their own scale. The reward weights stay
    untouched: dividing by a running standard deviation rescales the critic's
    target without changing which policy is optimal, since a positive affine
    rescaling of the return leaves the argmax and the ordering of advantages
    intact.

    The count is kept as a float and never reset, so the estimate is stable
    across a run whose reward scale drifts as the policy improves.
    """

    __slots__ = ("mean", "var", "count")

    def __init__(self, epsilon: float = 1e-4) -> None:
        self.mean = 0.0
        self.var = 1.0
        # Seeded rather than zero: the first update would otherwise divide by a
        # variance estimated from a single batch.
        self.count = epsilon

    def update(self, x: Tensor) -> None:
        batch_count = x.numel()
        if batch_count == 0:
            return
        batch_mean = float(x.mean().item())
        batch_var = float(x.var(unbiased=False).item())

        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta * delta * self.count * batch_count / total) / total
        self.count = total

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var))


def calculate_huber_loss(td_errors: Tensor, kappa: float = 1.0) -> Tensor:
    """Element-wise Huber loss.

    This is the standard definition used in distributional RL papers.
    """
    abs_td = td_errors.abs()
    quadratic = 0.5 * td_errors.pow(2)
    linear = kappa * (abs_td - 0.5 * kappa)
    return torch.where(abs_td <= kappa, quadratic, linear)


def calc_fraction_loss(
    fz_expected: Tensor,
    fz_tau: Tensor,
    taus: Tensor,
    weights: Optional[Tensor] = None,
) -> Tensor:
    """Fraction proposal network (FPN) loss for FQF.

    Parameters
    ----------
    fz_expected:
        Quantiles for taus_ (shape: [B, N, 1])
    fz_tau:
        Quantiles for internal taus (shape: [B, N-1, 1])
    taus:
        Cumulative taus including 0 and 1 (shape: [B, N+1])
    weights:
        Optional PER importance sampling weights (shape: [B, 1] or [B])

    Returns
    -------
    Tensor
        Scalar loss.
    """
    # Following the FQF paper: build gradients encouraging monotonic taus.
    gradients1 = fz_tau - fz_expected[:, :-1]
    gradients2 = fz_tau - fz_expected[:, 1:]

    flag_1 = fz_tau > torch.cat([fz_expected[:, :1], fz_tau[:, :-1]], dim=1)
    flag_2 = fz_tau < torch.cat([fz_tau[:, 1:], fz_expected[:, -1:]], dim=1)

    gradients = (
        torch.where(flag_1, gradients1, -gradients1)
        + torch.where(flag_2, gradients2, -gradients2)
    )
    gradients = gradients.view(taus.shape[0], -1).detach()

    # taus[:, 1:-1] corresponds to internal fractions (excluding 0 and 1).
    inner_taus = taus[:, 1:-1]
    loss_per_sample = (gradients * inner_taus).sum(dim=1)

    if weights is not None:
        weights = weights.view(-1)
        return (loss_per_sample * weights).mean()

    return loss_per_sample.mean()


# -----------------------------
# DQN
# -----------------------------
class DQNAgent:
    """Dueling DQN agent with optional PER and ICM curiosity."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        layer_type: str,
        layer_size: int,
        num_layers: int,
        use_batchnorm: bool,
        am: bool,
        n_steps: int,
        batch_size: int,
        buffer_size: int,
        lr: float,
        lr_dec: int,
        tau: float,
        gamma: float,
        munchausen: bool,
        curiosity: int,
        curiosity_size: int,
        per: int,
        rdm: int,
        entropy_tau: float,
        entropy_tau_coeff: float,
        lo: float,
        alpha: float,
        n_quantiles: int,
        entropy_coeff: float,
        update_every: int,
        max_train_steps: int,
        decay_update: int,
        device: torch.device,
        seed: int,
    ) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.layer_type = layer_type
        self.layer_size = layer_size
        self.num_layers = num_layers
        self.use_batchnorm = use_batchnorm
        self.am = am

        self.device = device
        self.seed = seed
        torch.manual_seed(seed)

        self.tau = tau
        self.gamma = gamma
        self.update_every = update_every
        self.t_step = 0

        self.batch_size = batch_size
        self.n_steps = n_steps

        # Bookkeeping
        self.q_updates = 1  # kept to match your "decay_update" convention

        # Optimizer / LR schedule config
        self.lr = lr
        self.lr_dec = lr_dec
        self.max_train_steps = max_train_steps
        self.decay_update = decay_update

        self.per = per
        self.rdm = rdm

        # Munchausen (not used in this DQN implementation, but kept for signature parity)
        self.munchausen = munchausen
        self.entropy_tau = entropy_tau
        self.entropy_tau_coeff = entropy_tau_coeff
        self.lo = lo
        self.alpha = alpha

        # ICM curiosity
        self.curiosity = curiosity
        self.curiosity_size = curiosity_size
        self.eta = 0.1  # intrinsic reward scale

        self.grad_clip = 1.0

        print(
            "lr decay:",
            self.lr_dec,
            "decay_update:",
            self.decay_update,
            "PER",
            self.per,
        )
        print("with AM" if self.am else "without AM")

        # Q-Networks
        self.qnetwork_local = Dueling_QNetwork(
            state_size,
            action_size,
            layer_size,
            n_steps,
            seed,
            num_layers,
            layer_type,
            use_batchnorm,
        ).to(device)
        self.qnetwork_target = Dueling_QNetwork(
            state_size,
            action_size,
            layer_size,
            n_steps,
            seed,
            num_layers,
            layer_type,
            use_batchnorm,
        ).to(device)

        # Optimizer
        if self.lr_dec == 0:
            self.optimizer = schedulefree.AdamWScheduleFree(
                self.qnetwork_local.parameters(), lr=lr
            )
            print("Schedule Free Optimizer")
        else:
            self.optimizer = optim.AdamW(self.qnetwork_local.parameters(), lr=lr)

        print(self.qnetwork_local)

        # Replay memory
        if self.per == 0:
            self.memory = ReplayBuffer(buffer_size, batch_size, seed, gamma, n_steps, rdm)
        elif self.per == 1:
            self.memory = PrioritizedReplay(buffer_size, batch_size, seed, gamma, n_steps)
        elif self.per == 2:
            self.memory = N_Steps_Prioritized_ReplayBuffer(
                buffer_size, batch_size, seed, gamma, n_steps
            )
        else:
            raise ValueError(f"Unsupported PER mode: {self.per}")

        # Curiosity module
        self.icm: Optional[ICM] = None
        if self.curiosity != 0:
            inverse_m = Inverse(self.state_size, self.action_size, self.curiosity_size)
            forward_m = Forward(
                self.state_size,
                self.action_size,
                inverse_m.calc_input_layer(),
                device=device,
            )
            self.icm = ICM(inverse_m, forward_m).to(device)
            print(inverse_m, forward_m)

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store one transition. No gradient work happens here."""
        state_t = torch.from_numpy(state).float()
        next_state_t = torch.from_numpy(next_state).float()

        self.memory.add(state_t, action, reward, next_state_t, done)
        self.t_step += 1

    def train_step(self) -> Optional[float]:
        """Learn from a batch if one is due, else None.

        Split from `observe` so the two can run at different rates or in
        different threads: an actor stepping the environment only needs to
        store, and whoever owns the network decides when to consume the
        buffer. `step` keeps the combined behaviour for the single-actor path.
        """
        if self.t_step % self.update_every != 0:
            return None

        if len(self.memory) <= self.batch_size:
            return None

        experiences = self.memory.sample()
        if self.per == 0:
            loss, _icm_loss = self.learn(experiences)
            loss_value = float(loss)
        else:
            loss_value = float(self.learn_per(experiences))

        self.q_updates += 1
        return loss_value

    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Optional[float]:
        """Store transition and trigger a learning step periodically."""
        self.observe(state, action, reward, next_state, done)
        return self.train_step()

    def act(
        self,
        state: Union[np.ndarray, Tensor],
        all_ff_waiting,
        eps: float = 0.0,
    ) -> Tuple[int, int, List[int]]:
        """Epsilon-greedy action selection with masking of invalid actions.

        Returns the feasible actions alongside the choice: callers use them for
        explainability (which alternatives were available and rejected).
        """
        potential_actions, potential_skills = get_potential_actions(state, all_ff_waiting)

        if np.random.uniform() <= eps:
            action = random.choice(potential_actions)
            skill_lvl = potential_skills[potential_actions.index(action)]
            return action, skill_lvl, potential_actions

        state_t = _as_tensor(state, self.device)
        with torch.inference_mode(), inference_pass(self.qnetwork_local):
            q = self.qnetwork_local(state_t)

        # Only the feasible actions are ever read, so the argmax is taken on
        # the device and one scalar comes back instead of the whole 80-value
        # vector. `.cpu()` was 3.2 s of a 60 s run across 33k transfers.
        action = _best_feasible_action(q, potential_actions)
        skill_lvl = potential_skills[potential_actions.index(action)]
        return action, skill_lvl, potential_actions

    def soft_update(self, local_model: torch.nn.Module, target_model: torch.nn.Module) -> None:
        """Polyak averaging update: target <- tau * local + (1-tau) * target."""
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )

    def learn(self, experiences):
        """Standard DQN learning step (no PER)."""
        icm_loss_value = 0.0

        self.optimizer.zero_grad()

        states, actions, rewards, next_states, dones = experiences
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(
            np.float32(next_states), dtype=torch.float32, device=self.device
        )
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Curiosity / intrinsic reward
        if self.curiosity != 0 and self.icm is not None:
            forward_err, inverse_err = self.icm.calc_errors(
                state1=states_t, state2=next_states_t, action=actions_t
            )
            r_i = self.eta * forward_err
            if r_i.shape != rewards_t.shape:
                raise ValueError("Intrinsic reward and extrinsic reward shapes mismatch.")

            if self.curiosity == 1:
                rewards_t = rewards_t + r_i.detach()
            else:
                rewards_t = r_i.detach()

            icm_loss_value = float(self.icm.update_ICM(forward_err, inverse_err))

        # Target: r + gamma^n * max_a' Q_target(s',a')
        q_targets_next = self.qnetwork_target(next_states_t).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards_t + (self.gamma**self.n_steps) * q_targets_next * (1.0 - dones_t)

        # Prediction: Q_local(s,a)
        q_expected = self.qnetwork_local(states_t).gather(1, actions_t)

        loss = F.mse_loss(q_expected, q_targets)
        loss.backward()
        clip_grad_norm_(self.qnetwork_local.parameters(), self.grad_clip)

        if self.lr_dec != 0:
            self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target)

        if self.q_updates % self.decay_update == 0:
            self._maybe_decay_lr()

        return loss.detach().cpu().item(), icm_loss_value

    def learn_per(self, experiences) -> float:
        """DQN learning step with PER sampling weights."""
        self.optimizer.zero_grad()

        states, actions, rewards, next_states, dones, idx, weights = experiences

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        weights_t = torch.as_tensor(weights, dtype=torch.float32, device=self.device).view(-1, 1)

        q_targets_next = self.qnetwork_target(next_states_t).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards_t + (self.gamma**self.n_steps) * q_targets_next * (1.0 - dones_t)

        # NOTE: This was `.gather(0, actions)` in the original file; that is almost
        # always wrong for DQN (dim=1 is the action dimension).
        q_expected = self.qnetwork_local(states_t).gather(1, actions_t)

        td_error = q_targets - q_expected
        loss = (td_error.pow(2) * weights_t).mean()

        loss.backward()
        clip_grad_norm_(self.qnetwork_local.parameters(), self.grad_clip)

        if self.lr_dec != 0:
            self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target)

        if self.q_updates % self.decay_update == 0:
            self._maybe_decay_lr()

        # Update PER priorities. Flatten the trailing axis: the sum-tree stores
        # one scalar per index, and a (batch, 1) array would hand it 1-element
        # arrays instead.
        self.memory.update_priorities(idx, td_error.detach().abs().flatten().cpu().numpy())
        return float(loss.detach().cpu().item())

    def _maybe_decay_lr(self) -> None:
        """Apply one of the LR decay modes used in this project."""
        print("update lr decay")
        if self.lr_dec == 0:
            self.lr_decay_0()
        elif self.lr_dec == 1:
            self.lr_decay_1()
        elif self.lr_dec == 2:
            self.lr_decay_2()
        elif self.lr_dec == 3:
            self.lr_decay_3()

    def lr_decay_0(self) -> None:
        lr_now = self.optimizer.param_groups[0]["lr"]
        print("step", self.t_step, "current lr :", lr_now)

    def lr_decay_1(self) -> None:
        lr_now = 0.9 * self.lr * (1 - self.t_step / self.max_train_steps) + 0.1 * self.lr
        for group in self.optimizer.param_groups:
            group["lr"] = lr_now
        print("step", self.t_step, "current lr :", lr_now)

    def lr_decay_2(self) -> None:
        if self.t_step % 5000 == 0:
            self.lr = self.lr / 2
            for group in self.optimizer.param_groups:
                group["lr"] = self.lr
        print("step", self.t_step, "current lr :", self.lr)

    def lr_decay_3(self) -> None:
        self.lr = self.lr / 2
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr
        print("step", self.t_step, "current lr :", self.lr)


# -----------------------------
# FQF
# -----------------------------
class FQFAgent:
    """FQF agent (Quantile Network + Fraction Proposal Network) with optional Munchausen and PER."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        layer_type: str,
        layer_size: int,
        num_layers: int,
        use_batchnorm: bool,
        am: bool,
        n_steps: int,
        batch_size: int,
        buffer_size: int,
        lr: float,
        lr_dec: int,
        tau: float,
        gamma: float,
        munchausen: bool,
        curiosity: int,
        curiosity_size: int,
        per: int,
        rdm: int,
        entropy_tau: float,
        entropy_tau_coeff: float,
        lo: float,
        alpha: float,
        n_quantiles: int,
        entropy_coeff: float,
        update_every: int,
        max_train_steps: int,
        decay_update: int,
        device: torch.device,
        seed: int,
        pointer: bool = False,
    ) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.layer_type = layer_type
        self.layer_size = layer_size
        self.num_layers = num_layers
        self.use_batchnorm = use_batchnorm
        self.am = am

        self.device = device
        self.seed = seed
        torch.manual_seed(seed)

        self.tau = tau
        self.gamma = gamma
        self.update_every = update_every
        self.t_step = 0

        self.batch_size = batch_size
        self.n_steps = n_steps

        self.entropy_coeff = entropy_coeff
        self.n_quantiles = n_quantiles

        self.lr = lr
        self.lr_dec = lr_dec
        self.max_train_steps = max_train_steps
        self.decay_update = decay_update

        self.per = per
        self.rdm = rdm

        self.munchausen = munchausen
        self.entropy_tau = entropy_tau
        self.entropy_tau_coeff = entropy_tau_coeff
        self.lo = lo
        self.alpha = alpha

        self.curiosity = curiosity
        self.curiosity_size = curiosity_size
        self.eta = 0.1

        self.grad_clip = 1.0
        self.q_updates = 1

        print(
            "lr decay:",
            self.lr_dec,
            "decay_update:",
            self.decay_update,
            "PER",
            self.per,
        )

        # Networks
        self.pointer = pointer
        qvn_args = (
            state_size, action_size, layer_size, am, n_steps, device, seed,
            n_quantiles, num_layers, layer_type, use_batchnorm,
        )
        self.qnetwork_local = QVN(*qvn_args, pointer=pointer).to(device)
        self.qnetwork_target = QVN(*qvn_args, pointer=pointer).to(device)
        self.optimizer = optim.AdamW(self.qnetwork_local.parameters(), lr=lr)
        print(self.qnetwork_local)

        self.fpn = FPN(layer_size, seed, n_quantiles, device).to(device)
        print(self.fpn)
        self.frac_optimizer = optim.RMSprop(
            self.fpn.parameters(), lr=lr * 1e-6, alpha=0.95, eps=1e-5
        )

        # Replay memory
        if self.per == 0:
            self.memory = ReplayBuffer(buffer_size, batch_size, seed, gamma, n_steps, rdm)
        elif self.per == 1:
            self.memory = PrioritizedReplay(buffer_size, batch_size, seed, gamma, n_steps)
        elif self.per == 2:
            self.memory = N_Steps_Prioritized_ReplayBuffer(
                buffer_size, batch_size, seed, gamma, n_steps
            )
        else:
            raise ValueError(f"Unsupported PER mode: {self.per}")

        # Curiosity module
        self.icm: Optional[ICM] = None
        if self.curiosity != 0:
            inverse_m = Inverse(self.state_size, self.action_size, self.curiosity_size)
            forward_m = Forward(
                self.state_size,
                self.action_size,
                inverse_m.calc_input_layer(),
                device=device,
            )
            self.icm = ICM(inverse_m, forward_m).to(device)
            print(inverse_m, forward_m)

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store one transition. No gradient work happens here."""
        self.memory.add(
            torch.from_numpy(state).float(),
            action,
            reward,
            torch.from_numpy(next_state).float(),
            done,
        )
        self.t_step += 1

    def train_step(self) -> Optional[float]:
        """Learn from a batch if one is due, else None.

        Split from `observe` so the two can run at different rates or in
        different threads: an actor stepping the environment only needs to
        store, and whoever owns the network decides when to consume the
        buffer. `step` keeps the combined behaviour for the single-actor path.
        """
        if self.t_step % self.update_every != 0:
            return None

        if len(self.memory) <= self.batch_size:
            return None

        experiences = self.memory.sample()
        if self.per == 0:
            loss, _entropy, _icm_loss = self.learn(experiences)
            loss_value = float(loss)
        else:
            loss_value, _entropy = self.learn_per(experiences)

        self.q_updates += 1
        return loss_value

    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Optional[float]:
        """Store transition and trigger a learning step periodically."""
        self.observe(state, action, reward, next_state, done)
        return self.train_step()

    def act(
        self,
        state: np.ndarray,
        all_ff_waiting,
        eps: float = 0.0,
        explain: Optional[dict] = None,
    ) -> Tuple[int, int, List[int]]:
        """Epsilon-greedy action selection based on expected value over quantiles.

        Returns the feasible actions alongside the choice: callers use them for
        explainability (which alternatives were available and rejected).

        `explain`, when given, is filled in place with the quantities this
        method computes and would otherwise discard: the Q-value of every
        feasible action, and the quantiles of the return distribution behind
        them. They are the numbers the choice was actually made from, so a log
        built on them explains the decision rather than approximating it. The
        return value is unchanged either way, so callers that pass nothing are
        unaffected.
        """
        potential_actions, potential_skills = get_potential_actions(state, all_ff_waiting)

        if np.random.uniform() <= eps:
            action = random.choice(potential_actions)
            skill_lvl = potential_skills[potential_actions.index(action)]
            if explain is not None:
                # An exploratory action has no Q behind it; say so rather than
                # let a reader mistake the greedy values for the reason.
                explain.update(random=True, q_values=None, quantiles=None)
            return action, skill_lvl, potential_actions

        state_t = torch.from_numpy(state.flatten()).float().to(self.device)

        with torch.inference_mode(), inference_pass(self.qnetwork_local):
            embedding = self.qnetwork_local.forward(state_t)
            entities = None
            if self.pointer:
                # The pointer path returns the candidate embeddings alongside
                # the context; only the context feeds the fraction network.
                embedding, entities = embedding
            taus, taus_, _entropy = self.fpn(embedding)
            f_z = self.qnetwork_local.get_quantiles(
                state_t, taus_, embedding, entities=entities
            )
            q = ((taus[:, 1:].unsqueeze(-1) - taus[:, :-1].unsqueeze(-1)) * f_z).sum(1)

        # The full vector is only needed to fill `explain`; the choice itself
        # reads the feasible actions alone, so the transfer is skipped when
        # nothing will report it.
        if explain is not None:
            q_list = q.detach().cpu().numpy().flatten().tolist()
            action = filter_q_values(q_list, potential_actions)
        else:
            action = _best_feasible_action(q, potential_actions)
        skill_lvl = potential_skills[potential_actions.index(action)]

        if explain is not None:
            explain.update(
                random=False,
                q_values={int(a): float(q_list[a]) for a in potential_actions},
                # Per-action return distribution. FQF models the whole
                # distribution and then averages it away; two actions with the
                # same mean can carry very different downside, which is the
                # distinction that matters when the cost is a coverage gap.
                quantiles={
                    int(a): f_z[0, :, a].detach().cpu().numpy().astype("float32")
                    for a in potential_actions
                },
            )

        return action, skill_lvl, potential_actions

    def _encode(self, network, states):
        """`(context, entities)` from a forward, whichever head is in use.

        The pointer path returns both; the flatten path returns the context
        alone and scores from slot position, so its entities are None. Every
        `get_quantiles` call in `learn`/`learn_per` goes through this, so the
        two paths differ in one place rather than at ten call sites.
        """
        out = network.forward(states)
        if self.pointer:
            return out
        return out, None

    def soft_update(self, local_model: torch.nn.Module, target_model: torch.nn.Module) -> None:
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )

    def learn(self, experiences):
        """FQF learning step without PER."""
        states, actions, rewards, next_states, dones = experiences

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(
            np.float32(next_states), dtype=torch.float32, device=self.device
        )
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Fraction proposal network produces taus and their midpoints taus_
        embedding, entities = self._encode(self.qnetwork_local, states_t)
        taus, taus_, entropy = self.fpn(embedding.detach())

        # Quantiles for current state-action
        f_z_expected = self.qnetwork_local.get_quantiles(
            states_t, taus_, embedding, entities=entities
        )
        q_expected = f_z_expected.gather(
            2, actions_t.unsqueeze(-1).expand(self.batch_size, self.n_quantiles, 1)
        )
        assert q_expected.shape == (self.batch_size, self.n_quantiles, 1)

        # Fraction loss
        with torch.inference_mode():
            f_z_tau = self.qnetwork_local.get_quantiles(
                states_t, taus[:, 1:-1], embedding.detach(), entities=entities
            )
            fz_tau = f_z_tau.gather(
                2, actions_t.unsqueeze(-1).expand(self.batch_size, self.n_quantiles - 1, 1)
            )

        frac_loss = calc_fraction_loss(q_expected.detach(), fz_tau, taus)
        frac_loss = frac_loss + self.entropy_coeff * entropy.mean()

        # Curiosity (optional): adds intrinsic reward
        icm_loss_value = 0.0
        if self.curiosity != 0 and self.icm is not None:
            forward_err, inverse_err = self.icm.calc_errors(
                state1=states_t, state2=next_states_t, action=actions_t
            )
            r_i = self.eta * forward_err
            if self.curiosity == 1:
                rewards_t = rewards_t + r_i.detach()
            else:
                rewards_t = r_i.detach()
            icm_loss_value = float(self.icm.update_ICM(forward_err, inverse_err))

        # Targets
        if not self.munchausen:
            with torch.inference_mode():
                next_embedding_loc, next_entities_loc = self._encode(
                    self.qnetwork_local, next_states_t
                )
                n_taus, n_taus_, _ = self.fpn(next_embedding_loc)
                f_z_next_loc = self.qnetwork_local.get_quantiles(
                    next_states_t, n_taus_, next_embedding_loc,
                    entities=next_entities_loc,
                )
                q_targets_next_loc = (
                    (n_taus[:, 1:].unsqueeze(-1) - n_taus[:, :-1].unsqueeze(-1))
                    * f_z_next_loc
                ).sum(1)
                action_idx = torch.argmax(q_targets_next_loc, dim=1, keepdim=True)

                next_embedding, next_entities = self._encode(
                    self.qnetwork_target, next_states_t
                )
                f_z_next = self.qnetwork_target.get_quantiles(
                    next_states_t, taus_, next_embedding, entities=next_entities
                )
                q_targets_next = (
                    f_z_next.gather(
                        2,
                        action_idx.unsqueeze(-1).expand(self.batch_size, self.n_quantiles, 1),
                    )
                    .transpose(1, 2)
                )
                q_targets = rewards_t.unsqueeze(-1) + (self.gamma**self.n_steps) * q_targets_next * (
                    1.0 - dones_t.unsqueeze(-1)
                )
        else:
            # Munchausen target (kept close to your original implementation)
            ns_embedding, ns_entities = self._encode(
                self.qnetwork_target, next_states_t
            )
            ns_embedding = ns_embedding.detach()
            ns_taus, ns_taus_, ns_entropy = self.fpn(ns_embedding.detach())
            ns_taus = ns_taus.detach()
            ns_entropy = ns_entropy.detach()

            m_quantiles = self.qnetwork_target.get_quantiles(
                next_states_t, ns_taus_, ns_embedding, entities=ns_entities
            ).detach()
            m_q = ((ns_taus[:, 1:].unsqueeze(-1) - ns_taus[:, :-1].unsqueeze(-1)) * m_quantiles).sum(
                1
            )

            logsum = torch.logsumexp(
                (m_q - m_q.max(1)[0].unsqueeze(-1))
                / (ns_entropy * self.entropy_tau_coeff).mean().detach(),
                1,
            ).unsqueeze(-1)
            tau_log_pi_next = (
                m_q
                - m_q.max(1)[0].unsqueeze(-1)
                - (ns_entropy * self.entropy_tau_coeff).mean().detach() * logsum
            ).unsqueeze(1)

            pi_target = F.softmax(
                m_q / (ns_entropy * self.entropy_tau_coeff).mean().detach(), dim=1
            ).unsqueeze(1)
            q_target = (
                (self.gamma**self.n_steps)
                * (pi_target * (m_quantiles - tau_log_pi_next) * (1 - dones_t.unsqueeze(-1))).sum(2)
            ).unsqueeze(1)

            m_quantiles_targets = self.qnetwork_local.get_quantiles(
                states_t, taus_, embedding, entities=entities
            ).detach()
            m_q_targets = (
                (taus[:, 1:].unsqueeze(-1).detach() - taus[:, :-1].unsqueeze(-1).detach())
                * m_quantiles_targets
            ).sum(1)
            v_k_target = m_q_targets.max(1)[0].unsqueeze(-1)
            tau_log_pik = (
                m_q_targets
                - v_k_target
                - (entropy * self.entropy_tau_coeff).mean().detach()
                * torch.logsumexp(
                    (m_q_targets - v_k_target)
                    / (entropy * self.entropy_tau_coeff).mean().detach(),
                    1,
                ).unsqueeze(-1)
            )
            munchausen_addon = tau_log_pik.gather(1, actions_t)
            munchausen_reward = (
                rewards_t + self.alpha * torch.clamp(munchausen_addon, min=self.lo, max=0.0)
            ).unsqueeze(-1)

            q_targets = munchausen_reward + q_target

        # Quantile Huber loss
        td_error = q_targets - q_expected
        huber_l = calculate_huber_loss(td_error, kappa=1.0)
        quantile_l = (
            (taus_.unsqueeze(-1) - (td_error.detach() < 0).float()).abs() * huber_l
        )
        loss = quantile_l.sum(dim=1).mean(dim=1).mean()

        # Optimize FPN first (retain graph because loss uses shared tensors)
        self.frac_optimizer.zero_grad()
        frac_loss.backward(retain_graph=True)
        self.frac_optimizer.step()

        # Optimize Q network
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.qnetwork_local.parameters(), self.grad_clip)
        if self.lr_dec != 0:
            self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target)

        if self.q_updates % self.decay_update == 0:
            self._maybe_decay_lr()

        return float(loss.detach().cpu().item()), float(entropy.mean().detach().cpu().item()), icm_loss_value

    def learn_per(self, experiences):
        """FQF learning step with PER."""
        states, actions, rewards, next_states, dones, idx, weights = experiences

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(
            np.float32(next_states), dtype=torch.float32, device=self.device
        )
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        weights_t = torch.as_tensor(weights, dtype=torch.float32, device=self.device).view(-1, 1)

        embedding, entities = self._encode(self.qnetwork_local, states_t)
        taus, taus_, entropy = self.fpn(embedding.detach())

        f_z_expected = self.qnetwork_local.get_quantiles(
            states_t, taus_, embedding, entities=entities
        )
        q_expected = f_z_expected.gather(
            2, actions_t.unsqueeze(-1).expand(self.batch_size, self.n_quantiles, 1)
        )

        with torch.inference_mode():
            f_z_tau = self.qnetwork_local.get_quantiles(
                states_t, taus[:, 1:-1], embedding.detach(), entities=entities
            )
            fz_tau = f_z_tau.gather(
                2, actions_t.unsqueeze(-1).expand(self.batch_size, self.n_quantiles - 1, 1)
            )

        frac_loss = calc_fraction_loss(q_expected.detach(), fz_tau, taus, weights=weights_t)
        frac_loss = frac_loss + self.entropy_coeff * entropy.mean()

        # Targets (munchausen or not) - keep original logic
        if not self.munchausen:
            with torch.inference_mode():
                next_embedding_loc, next_entities_loc = self._encode(
                    self.qnetwork_local, next_states_t
                )
                n_taus, n_taus_, _ = self.fpn(next_embedding_loc)
                f_z_next_loc = self.qnetwork_local.get_quantiles(
                    next_states_t, n_taus_, next_embedding_loc,
                    entities=next_entities_loc,
                )
                q_targets_next_loc = (
                    (n_taus[:, 1:].unsqueeze(-1) - n_taus[:, :-1].unsqueeze(-1))
                    * f_z_next_loc
                ).sum(1)
                action_idx = torch.argmax(q_targets_next_loc, dim=1, keepdim=True)

                next_embedding, next_entities = self._encode(
                    self.qnetwork_target, next_states_t
                )
                f_z_next = self.qnetwork_target.get_quantiles(
                    next_states_t, taus_, next_embedding, entities=next_entities
                )
                q_targets_next = (
                    f_z_next.gather(
                        2,
                        action_idx.unsqueeze(-1).expand(self.batch_size, self.n_quantiles, 1),
                    )
                    .transpose(1, 2)
                )
                q_targets = rewards_t.unsqueeze(-1) + (self.gamma**self.n_steps) * q_targets_next * (
                    1.0 - dones_t.unsqueeze(-1)
                )
        else:
            ns_embedding, ns_entities = self._encode(
                self.qnetwork_target, next_states_t
            )
            ns_embedding = ns_embedding.detach()
            ns_taus, ns_taus_, ns_entropy = self.fpn(ns_embedding.detach())
            ns_taus = ns_taus.detach()
            ns_entropy = ns_entropy.detach()

            m_quantiles = self.qnetwork_target.get_quantiles(
                next_states_t, ns_taus_, ns_embedding, entities=ns_entities
            ).detach()
            m_q = ((ns_taus[:, 1:].unsqueeze(-1) - ns_taus[:, :-1].unsqueeze(-1)) * m_quantiles).sum(
                1
            )

            logsum = torch.logsumexp(
                (m_q - m_q.max(1)[0].unsqueeze(-1))
                / (ns_entropy * self.entropy_tau_coeff).mean().detach(),
                1,
            ).unsqueeze(-1)
            tau_log_pi_next = (
                m_q
                - m_q.max(1)[0].unsqueeze(-1)
                - (ns_entropy * self.entropy_tau_coeff).mean().detach() * logsum
            ).unsqueeze(1)

            pi_target = F.softmax(
                m_q / (ns_entropy * self.entropy_tau_coeff).mean().detach(), dim=1
            ).unsqueeze(1)
            q_target = (
                (self.gamma**self.n_steps)
                * (pi_target * (m_quantiles - tau_log_pi_next) * (1 - dones_t.unsqueeze(-1))).sum(2)
            ).unsqueeze(1)

            m_quantiles_targets = self.qnetwork_local.get_quantiles(
                states_t, taus_, embedding, entities=entities
            ).detach()
            m_q_targets = (
                (taus[:, 1:].unsqueeze(-1).detach() - taus[:, :-1].unsqueeze(-1).detach())
                * m_quantiles_targets
            ).sum(1)
            v_k_target = m_q_targets.max(1)[0].unsqueeze(-1)
            tau_log_pik = (
                m_q_targets
                - v_k_target
                - (entropy * self.entropy_tau_coeff).mean().detach()
                * torch.logsumexp(
                    (m_q_targets - v_k_target)
                    / (entropy * self.entropy_tau_coeff).mean().detach(),
                    1,
                ).unsqueeze(-1)
            )
            munchausen_addon = tau_log_pik.gather(1, actions_t)
            munchausen_reward = (
                rewards_t + self.alpha * torch.clamp(munchausen_addon, min=self.lo, max=0.0)
            ).unsqueeze(-1)

            q_targets = munchausen_reward + q_target

        td_error = q_targets - q_expected
        huber_l = calculate_huber_loss(td_error, kappa=1.0)
        quantile_l = (
            (taus_.unsqueeze(-1) - (td_error.detach() < 0).float()).abs() * huber_l
        )

        loss = (quantile_l.sum(dim=1).mean(dim=1, keepdim=True) * weights_t).mean()

        self.frac_optimizer.zero_grad()
        frac_loss.backward(retain_graph=True)
        self.frac_optimizer.step()

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.qnetwork_local.parameters(), self.grad_clip)
        if self.lr_dec != 0:
            self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target)

        if self.q_updates % self.decay_update == 0:
            self._maybe_decay_lr()

        # PER priorities: reduce td_error across quantiles to one value per
        # sample. keepdim would leave a trailing axis, so each "priority" would
        # be a 1-element array and the sum-tree assignment would fail.
        td_error_scalar = td_error.sum(dim=1).mean(dim=1)
        self.memory.update_priorities(idx, td_error_scalar.detach().abs().cpu().numpy())

        return float(loss.detach().cpu().item()), float(entropy.mean().detach().cpu().item())

    def _maybe_decay_lr(self) -> None:
        print("update lr decay")
        if self.lr_dec == 0:
            self.lr_decay_0()
        elif self.lr_dec == 1:
            self.lr_decay_1()
        elif self.lr_dec == 2:
            self.lr_decay_2()
        elif self.lr_dec == 3:
            self.lr_decay_3()

    def lr_decay_0(self) -> None:
        lr_now = self.optimizer.param_groups[0]["lr"]
        print("step", self.t_step, "current lr :", lr_now)

    def lr_decay_1(self) -> None:
        lr_now = 0.9 * self.lr * (1 - self.t_step / self.max_train_steps) + 0.1 * self.lr
        for group in self.optimizer.param_groups:
            group["lr"] = lr_now
        print("step", self.t_step, "current lr :", lr_now)

    def lr_decay_2(self) -> None:
        if self.t_step % 5000 == 0:
            self.lr = self.lr / 2
            for group in self.optimizer.param_groups:
                group["lr"] = self.lr
        print("step", self.t_step, "current lr :", self.lr)

    def lr_decay_3(self) -> None:
        self.lr = self.lr / 2
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr
        print("step", self.t_step, "current lr :", self.lr)


# -----------------------------
# Decision Transformer
# -----------------------------
class DTAgent:
    """Decision Transformer agent (policy conditioned on return-to-go)."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        feature_size: int,
        buffer_size: int,
        batch_size: int,
        update_every: int,
        num_layers: int,
        lr: float,
        layer_size: int,
        device: torch.device,
        max_len: int,
        seed: int,
    ) -> None:
        self.device = device
        self.state_size = state_size
        self.action_size = action_size
        self.feature_size = feature_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_every = update_every
        self.num_layers = num_layers
        self.layer_size = layer_size
        self.max_len = max_len
        self.seed = seed

        self.dt_network = DT_Network(
            self.state_size,
            self.action_size,
            self.feature_size,
            self.layer_size,
            self.num_layers,
            self.max_len,
            self.seed,
        ).to(device)

        # Optional compile for speed (PyTorch 2+).
        self.dt_network = torch.compile(self.dt_network, dynamic=True)

        self.optimizer = optim.Adam(self.dt_network.parameters(), lr=lr)
        self.memory = DT_ReplayBuffer(self.buffer_size, self.batch_size)

    def act(
        self,
        state: np.ndarray,
        all_ff_waiting,
        traj_states: List[Tensor],
        traj_actions: List[Tensor],
        traj_returns: List[Tensor],
        traj_timesteps: List[int],
    ) -> Tuple[int, int, List[int]]:
        """Sample an action from the DT policy, masked to valid actions.

        Returns the feasible actions alongside the choice: callers use them for
        explainability (which alternatives were available and rejected).
        """
        potential_actions, potential_skills = get_potential_actions(state, all_ff_waiting)

        # Not enough history -> fallback to random valid action
        if not traj_states or not traj_actions or not traj_returns or not traj_timesteps:
            action = random.choice(potential_actions)
            skill_lvl = potential_skills[potential_actions.index(action)]
            return action, skill_lvl, potential_actions

        states = torch.stack(traj_states[-self.max_len :]).unsqueeze(0).to(self.device)
        actions = torch.stack(traj_actions[-self.max_len :]).unsqueeze(0).to(self.device)
        timesteps = torch.tensor(traj_timesteps[-self.max_len :], device=self.device).unsqueeze(0)

        returns_to_go = torch.stack(traj_returns[-self.max_len :]).unsqueeze(0).to(self.device)

        self.dt_network.eval()
        with torch.inference_mode():
            mask = torch.ones(states.shape[:2], dtype=torch.bool, device=self.device)
            logits = self.dt_network(states, actions, returns_to_go, timesteps, mask)
            last_logits = logits[:, -1]

            # Mask invalid actions by setting logits to -inf
            masked_logits = torch.full_like(last_logits, float("-inf"))
            masked_logits[:, potential_actions] = last_logits[:, potential_actions]

            probs = F.softmax(masked_logits, dim=-1)
            action = int(torch.multinomial(probs, num_samples=1).item())
        self.dt_network.train()

        skill_lvl = potential_skills[potential_actions.index(action)]
        return action, skill_lvl, potential_actions

    def store_trajectory(
        self,
        states: Tensor,
        actions: Tensor,
        returns_to_go: Tensor,
        timesteps: Tensor,
    ) -> None:
        self.memory.add((states, actions, returns_to_go, timesteps))

    def learn(self) -> Optional[float]:
        """Train DT on the last-step action prediction."""
        if len(self.memory) < self.memory.batch_size:
            return None

        states, actions, returns, timesteps, mask = self.memory.sample()
        states = states.to(self.device)
        actions = actions.to(self.device)
        returns = returns.to(self.device)
        timesteps = timesteps.to(self.device)
        mask = mask.to(self.device)

        self.optimizer.zero_grad()
        logits = self.dt_network(states, actions, returns, timesteps, mask)
        last_logits = logits[:, -1]  # predict action at last step

        targets = actions[:, -1]
        loss = F.cross_entropy(last_logits, targets)
        loss.backward()
        clip_grad_norm_(self.dt_network.parameters(), 1.0)
        self.optimizer.step()
        return float(loss.item())


# -----------------------------
# PPO
# -----------------------------
class PPOAgent:
    """PPO actor-critic with action masking and optional curiosity.

    Implements the clipped surrogate objective of Schulman et al. (2017):
    GAE(lambda) advantages bootstrapped at the rollout boundary, `n_epochs`
    minibatch passes over each rollout, and a ratio clipped to
    `1 +/- clip_range` with an approximate-KL early stop.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        layer_type: str,
        layer_size: int,
        num_layers: int,
        use_batchnorm: bool,
        am: bool,
        n_steps: int,
        batch_size: int,
        buffer_size: int,
        lr: float,
        lr_dec: int,
        tau: float,
        gamma: float,
        munchausen: bool,
        curiosity: int,
        curiosity_size: int,
        per: int,
        rdm: int,
        entropy_tau: float,
        entropy_tau_coeff: float,
        lo: float,
        alpha: float,
        n_quantiles: int,
        entropy_coeff: float,
        update_every: int,
        max_train_steps: int,
        decay_update: int,
        device: torch.device,
        seed: int,
        clip_range: float = 0.2,
        n_epochs: int = 10,
        minibatch_size: int = 64,
        gae_lambda: float = 0.95,
        value_coeff: float = 0.5,
        target_kl: Optional[float] = 0.03,
        normalize_returns: bool = True,
    ) -> None:
        self.am = am

        # Actor/critic LRs derive from the configured `lr` so the hyper-parameter
        # file is actually honoured: the decay schedules below already write
        # `self.lr` into both optimizers, so hard-coding the initial values made
        # the first `decay_update` steps run at a rate nothing had asked for.
        self.actor_lr = lr
        self.critic_lr = lr

        # Return normalisation. The reward weights are left alone -- the scale is
        # divided out here instead, where it actually hurts: the critic's target
        # is a discounted sum of -100 penalties (roughly -600 at gamma 0.99) and
        # it starts from zero, so the squared error and its gradient dwarf the
        # policy term. FQF is spared this by its Huber loss, whose gradient is
        # clipped past kappa; the PPO critic uses a plain MSE and is not.
        self.normalize_returns = normalize_returns
        self.return_rms = RunningMeanStd()

        # PPO surrogate objective.
        self.clip_range = clip_range
        self.n_epochs = n_epochs
        self.minibatch_size = minibatch_size
        self.gae_lambda = gae_lambda
        self.value_coeff = value_coeff
        # Early-stops the epoch loop once the updated policy has drifted this
        # far from the behaviour policy. None disables the check.
        self.target_kl = target_kl

        # Rollout storage: one entry per decision, holding the behaviour
        # policy's log-prob and value alongside the transition. PPO's ratio is
        # meaningless without the log-prob recorded *at acting time*: recomputing
        # it after the update would give a ratio of exactly 1 on the first epoch
        # and silently disable clipping.
        self.rollout_storage = []
        self.last_invalid_actions: Optional[List[int]] = None
        self.last_log_prob: Optional[float] = None
        self.last_value: Optional[float] = None

        self.state_size = state_size
        self.action_size = action_size
        self.layer_type = layer_type
        self.layer_size = layer_size
        self.num_layers = num_layers
        self.use_batchnorm = use_batchnorm
        self.seed = seed
        torch.manual_seed(seed)

        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size

        # LR schedule parameters (for parity with other agents)
        self.lr = lr
        self.lr_dec = lr_dec
        self.max_train_steps = max_train_steps
        self.decay_update = decay_update
        self.q_updates = 1

        self.entropy_coeff = entropy_coeff

        # Munchausen-style reward shaping (optional)
        self.munchausen = munchausen
        self.lo = lo
        self.alpha = alpha

        # Curiosity module (optional)
        self.curiosity = curiosity
        self.curiosity_size = curiosity_size
        self.eta = 0.1

        self.grad_clip = 1.0
        self.t_step = 0

        print(
            "lr decay:",
            self.lr_dec,
            "decay_update:",
            self.decay_update,
            "PER",
            per,
        )
        print("with AM" if self.am else "without AM")

        if self.am:
            self.model = PPO_ActorCriticAM(
                state_size,
                action_size,
                layer_size,
                seed,
                num_layers=num_layers,
                use_batchnorm=use_batchnorm,
            ).to(device)
        else:
            self.model = PPOActorCritic(
                state_size,
                action_size,
                layer_size,
                seed,
                num_layers=num_layers,
                use_batchnorm=use_batchnorm,
            ).to(device)

        # The runner, `checkpoint.save` and the model-saving path all reach for
        # `qnetwork_local`, which is the name the value-based agents give their
        # trained module. Binding the same object under that name -- not a copy --
        # lets `--train`/eval mode, weight loading and checkpointing work on PPO
        # without special-casing the agent everywhere.
        self.qnetwork_local = self.model

        # Curiosity module
        self.icm: Optional[ICM] = None
        if self.curiosity != 0:
            inverse_m = Inverse(self.state_size, self.action_size, self.curiosity_size)
            forward_m = Forward(
                self.state_size,
                self.action_size,
                inverse_m.calc_input_layer(),
                device=device,
            )
            self.icm = ICM(inverse_m, forward_m).to(device)

        # The bodies and heads are disjoint, but the attention block and the two
        # row encoders that feed them are shared. Listing only the bodies and
        # heads left those 25k parameters -- the attention that compares the 80
        # candidates against each other, which is the heart of the assignment
        # problem -- in no optimiser at all: they stayed at their random
        # initialisation for the whole run, and actor and critic had to learn on
        # top of a fixed arbitrary encoding of the state.
        #
        # They are attached to the actor, so the encoder is shaped by the policy
        # objective, and are deliberately *not* given to the critic as well:
        # a parameter in both optimisers would take two Adam steps per minibatch
        # off inconsistent moment estimates.
        shared_parameters = [
            p
            for name, p in self.model.named_parameters()
            if not name.startswith(
                ("actor_body", "critic_body", "policy_head", "value_head")
            )
        ]

        self.actor_parameters = (
            list(self.model.actor_body.parameters())
            + list(self.model.policy_head.parameters())
            + shared_parameters
        )
        self.critic_parameters = (
            list(self.model.critic_body.parameters()) + list(self.model.value_head.parameters())
        )

        self.actor_optimizer = optim.Adam(self.actor_parameters, lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_parameters, lr=self.critic_lr)

    def act(
        self,
        state: np.ndarray,
        all_ff_waiting,
        eps: float = 0.0,
        explain: Optional[dict] = None,
    ) -> Tuple[int, int, List[int]]:
        """Sample an action from the masked categorical distribution.

        Returns the feasible actions alongside the choice: callers use them for
        explainability (which alternatives were available and rejected).

        The sampled action's log-prob and the critic's value are stashed for the
        next `step`, because PPO's importance ratio is defined against the policy
        that actually collected the data.

        `explain`, when given, is filled in place with the per-action policy
        probabilities and the state value, mirroring the FQF agent so the
        decision log works for either.
        """
        potential_actions, potential_skills = get_potential_actions(state, all_ff_waiting)
        state_t = torch.from_numpy(state).float().to(self.device)

        with torch.inference_mode(), inference_pass(self.model):
            logits, value = self.model(state_t)

        logits = logits.squeeze(0)
        value = value.reshape(-1)[0]
        invalid_actions = [a for a in range(self.action_size) if a not in potential_actions]

        masked_logits = logits.clone()
        if invalid_actions:
            masked_logits[invalid_actions] = -1e9

        dist = torch.distributions.Categorical(logits=masked_logits)
        action_t = dist.sample()
        action = int(action_t.item())

        # `.clone()` because inference-mode tensors cannot be recorded and later
        # used in an autograd graph; these are stored, not differentiated.
        self.last_log_prob = float(dist.log_prob(action_t).item())
        self.last_value = float(value.item())
        self.last_invalid_actions = invalid_actions

        if explain is not None:
            probs = dist.probs
            explain["values"] = {a: float(probs[a].item()) for a in potential_actions}
            explain["state_value"] = self.last_value

        skill_lvl = potential_skills[potential_actions.index(action)]
        return action, skill_lvl, potential_actions

    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Optional[Tuple[Tensor, Tensor]]:
        """Store the rollout entry and trigger a PPO update once the batch is full."""
        state_t = torch.from_numpy(state).float()
        next_state_t = torch.from_numpy(next_state).float()

        invalid_actions = self.last_invalid_actions or []
        # A `step` without a preceding `act` (a resumed run, a scripted action)
        # has no behaviour log-prob. Recording 0.0 would claim probability 1 and
        # skew the ratio, so the entry is skipped instead.
        if self.last_log_prob is None:
            self.last_invalid_actions = None
            return None

        self.rollout_storage.append(
            (
                state_t,
                action,
                reward,
                next_state_t,
                done,
                invalid_actions,
                self.last_log_prob,
                self.last_value,
            )
        )
        self.last_invalid_actions = None
        self.last_log_prob = None
        self.last_value = None

        self.t_step += 1
        if self.t_step % self.batch_size == 0:
            return self.learn()

        return None

    def _bootstrap_value(self, next_state: Tensor, done: bool) -> float:
        """Value of the state the rollout was cut at.

        The rollout boundary is arbitrary -- it falls every `batch_size`
        decisions, not at an episode end -- so truncating the return there
        would treat an unfinished intervention as if it earned nothing more.
        A terminal state is worth 0 by definition and is not bootstrapped.
        """
        if done:
            return 0.0
        with torch.inference_mode(), inference_pass(self.model):
            _logits, value = self.model(next_state.unsqueeze(0).to(self.device))
        return float(value.reshape(-1)[0].item())

    def learn(self) -> Optional[Tuple[Tensor, Tensor]]:
        """Clipped-surrogate PPO update over the stored rollout.

        Differs from a plain policy gradient in the three ways that define PPO:
        advantages come from GAE(lambda) rather than raw Monte-Carlo returns,
        the rollout is reused for `n_epochs` passes in minibatches instead of a
        single pass, and the objective is clipped so those extra passes cannot
        walk the policy arbitrarily far from the data that justified the update.
        """
        if not self.rollout_storage:
            return None

        states = torch.stack([m[0] for m in self.rollout_storage]).to(self.device)
        actions = torch.tensor(
            [m[1] for m in self.rollout_storage], dtype=torch.long, device=self.device
        )
        rewards = torch.tensor(
            [m[2] for m in self.rollout_storage], dtype=torch.float32, device=self.device
        )
        next_states = torch.stack([m[3] for m in self.rollout_storage]).to(self.device)
        dones = [m[4] for m in self.rollout_storage]
        invalid_actions = [m[5] for m in self.rollout_storage]
        old_log_probs = torch.tensor(
            [m[6] for m in self.rollout_storage], dtype=torch.float32, device=self.device
        )
        values = torch.tensor(
            [m[7] for m in self.rollout_storage], dtype=torch.float32, device=self.device
        )

        # Optional curiosity augmentation
        if self.curiosity != 0 and self.icm is not None:
            forward_err, inverse_err = self.icm.calc_errors(
                state1=states, state2=next_states, action=actions.unsqueeze(1)
            )
            intrinsic_reward = self.eta * forward_err.squeeze(-1)
            if self.curiosity == 1:
                rewards = rewards + intrinsic_reward.detach()
            else:
                rewards = intrinsic_reward.detach()
            _icm_loss = self.icm.update_ICM(forward_err, inverse_err)

        # Munchausen shaping is a value-based (DQN) trick: it rewrites the reward
        # with a log-policy term that assumes a soft-greedy target. Under PPO it
        # biases the advantage the surrogate is built on, so it is refused rather
        # than silently applied.
        if self.munchausen:
            raise ValueError(
                "munchausen shaping is not compatible with the PPO surrogate; "
                "set 'munchausen': 0 in the PPO hyper-parameter file"
            )

        # --- Return normalisation ---
        # The reward weights are left untouched; the scale is divided out here.
        # The critic is trained on the normalised return below, so the values it
        # predicts -- those stored by `act` and the bootstrap -- are already on
        # the normalised scale. Only the raw rewards have to be divided, and the
        # GAE recursion below then mixes quantities that share one scale.
        #
        # The estimate is updated from the *raw* discounted return, so the
        # divisor tracks the true scale of the objective rather than the scale of
        # the already-normalised signal, which would collapse towards 1.
        if self.normalize_returns:
            with torch.no_grad():
                discounted = torch.zeros_like(rewards)
                running = 0.0
                for t in reversed(range(len(rewards))):
                    if dones[t]:
                        running = 0.0
                    running = float(rewards[t].item()) + self.gamma * running
                    discounted[t] = running
            self.return_rms.update(discounted)
            rewards = rewards / max(self.return_rms.std, 1e-6)

        # --- GAE(lambda) ---
        last_value = self._bootstrap_value(next_states[-1], dones[-1])
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_non_terminal = 0.0 if dones[t] else 1.0
            next_value = last_value if t == len(rewards) - 1 else float(values[t + 1].item())
            delta = (
                float(rewards[t].item())
                + self.gamma * next_value * next_non_terminal
                - float(values[t].item())
            )
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae

        # The critic regresses on the same quantity the advantage was measured
        # against, so returns are rebuilt from GAE rather than recomputed.
        returns = advantages + values

        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Masks are rebuilt once as a dense tensor: the epoch loop indexes it per
        # minibatch instead of walking the per-step Python lists n_epochs times.
        mask = torch.zeros(
            (len(invalid_actions), self.action_size), dtype=torch.bool, device=self.device
        )
        for i, inval in enumerate(invalid_actions):
            if inval:
                mask[i, inval] = True

        n = len(self.rollout_storage)
        minibatch_size = min(self.minibatch_size, n)
        actor_loss = torch.zeros((), device=self.device)
        critic_loss = torch.zeros((), device=self.device)
        stop_early = False

        for _epoch in range(self.n_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, minibatch_size):
                idx = perm[start : start + minibatch_size]
                # A trailing minibatch of one makes BatchNorm's batch statistics
                # undefined and the advantage std meaningless; fold it away.
                if idx.numel() < 2:
                    continue

                logits, value_pred = self.model(states[idx])
                masked_logits = logits.masked_fill(mask[idx], -1e9)

                dist = torch.distributions.Categorical(logits=masked_logits)
                log_probs = dist.log_prob(actions[idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(log_probs - old_log_probs[idx])
                adv = advantages[idx]
                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv
                actor_loss = -torch.min(unclipped, clipped).mean() - self.entropy_coeff * entropy

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                clip_grad_norm_(self.actor_parameters, self.grad_clip)
                self.actor_optimizer.step()

                # The critic shares no parameters with the actor here (separate
                # bodies and heads), but the value head was consumed by the graph
                # the actor step just freed, so it is re-evaluated.
                _logits, value_pred = self.model(states[idx])
                critic_loss = self.value_coeff * F.mse_loss(
                    value_pred.reshape(-1), returns[idx]
                )

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                clip_grad_norm_(self.critic_parameters, self.grad_clip)
                self.critic_optimizer.step()

            if self.target_kl is not None:
                with torch.inference_mode(), inference_pass(self.model):
                    logits, _value = self.model(states)
                    full_dist = torch.distributions.Categorical(
                        logits=logits.masked_fill(mask, -1e9)
                    )
                    approx_kl = (old_log_probs - full_dist.log_prob(actions)).mean().item()
                if approx_kl > self.target_kl:
                    stop_early = True

            if stop_early:
                break

        self.rollout_storage.clear()

        if self.q_updates % self.decay_update == 0:
            self._maybe_decay_lr()

        self.q_updates += 1
        return actor_loss.detach(), critic_loss.detach()

    def _maybe_decay_lr(self) -> None:
        """Apply LR decay to both actor and critic optimizers (if configured)."""
        print("update lr decay")
        if self.lr_dec == 0:
            self.lr_decay_0()
        elif self.lr_dec == 1:
            self.lr_decay_1()
        elif self.lr_dec == 2:
            self.lr_decay_2()
        elif self.lr_dec == 3:
            self.lr_decay_3()

    def lr_decay_0(self) -> None:
        actor_lr = self.actor_optimizer.param_groups[0]["lr"]
        critic_lr = self.critic_optimizer.param_groups[0]["lr"]
        print("step", self.t_step, "actor lr:", actor_lr, "critic lr:", critic_lr)

    def lr_decay_1(self) -> None:
        lr_now = 0.9 * self.lr * (1 - self.t_step / self.max_train_steps) + 0.1 * self.lr
        for group in self.actor_optimizer.param_groups:
            group["lr"] = lr_now
        for group in self.critic_optimizer.param_groups:
            group["lr"] = lr_now
        print("step", self.t_step, "current lr :", lr_now)

    def lr_decay_2(self) -> None:
        if self.t_step % 5000 == 0:
            self.lr = self.lr / 2
            for group in self.actor_optimizer.param_groups:
                group["lr"] = self.lr
            for group in self.critic_optimizer.param_groups:
                group["lr"] = self.lr
        print("step", self.t_step, "current lr :", self.lr)

    def lr_decay_3(self) -> None:
        self.lr = self.lr / 2
        for group in self.actor_optimizer.param_groups:
            group["lr"] = self.lr
        for group in self.critic_optimizer.param_groups:
            group["lr"] = self.lr
        print("step", self.t_step, "current lr :", self.lr)


# -----------------------------
# Backward compatible aliases
# -----------------------------
DQN_Agent = DQNAgent
FQF_Agent = FQFAgent
DT_Agent = DTAgent
PPO_Agent = PPOAgent
