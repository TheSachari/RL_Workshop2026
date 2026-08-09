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

    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Optional[float]:
        """Store transition and trigger a learning step periodically."""
        state_t = torch.from_numpy(state).float()
        next_state_t = torch.from_numpy(next_state).float()

        self.memory.add(state_t, action, reward, next_state_t, done)
        self.t_step += 1

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
        self.qnetwork_local.eval()
        with torch.inference_mode():
            q = self.qnetwork_local(state_t)
        self.qnetwork_local.train()

        q_values = q.detach().cpu().numpy().flatten().tolist()
        action = filter_q_values(q_values, potential_actions)
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
        self.qnetwork_local = QVN(
            state_size,
            action_size,
            layer_size,
            am,
            n_steps,
            device,
            seed,
            n_quantiles,
            num_layers,
            layer_type,
            use_batchnorm,
        ).to(device)
        self.qnetwork_target = QVN(
            state_size,
            action_size,
            layer_size,
            am,
            n_steps,
            device,
            seed,
            n_quantiles,
            num_layers,
            layer_type,
            use_batchnorm,
        ).to(device)
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

    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Optional[float]:
        """Store transition and trigger a learning step periodically."""
        self.memory.add(
            torch.from_numpy(state).float(),
            action,
            reward,
            torch.from_numpy(next_state).float(),
            done,
        )
        self.t_step += 1

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

    def act(
        self,
        state: np.ndarray,
        all_ff_waiting,
        eps: float = 0.0,
    ) -> Tuple[int, int, List[int]]:
        """Epsilon-greedy action selection based on expected value over quantiles.

        Returns the feasible actions alongside the choice: callers use them for
        explainability (which alternatives were available and rejected).
        """
        potential_actions, potential_skills = get_potential_actions(state, all_ff_waiting)

        if np.random.uniform() <= eps:
            action = random.choice(potential_actions)
            skill_lvl = potential_skills[potential_actions.index(action)]
            return action, skill_lvl, potential_actions

        state_t = torch.from_numpy(state.flatten()).float().to(self.device)

        self.qnetwork_local.eval()
        with torch.inference_mode():
            embedding = self.qnetwork_local.forward(state_t)
            taus, taus_, _entropy = self.fpn(embedding)
            f_z = self.qnetwork_local.get_quantiles(state_t, taus_, embedding)
            q = ((taus[:, 1:].unsqueeze(-1) - taus[:, :-1].unsqueeze(-1)) * f_z).sum(1)
        self.qnetwork_local.train()

        q_list = q.detach().cpu().numpy().flatten().tolist()
        action = filter_q_values(q_list, potential_actions)
        skill_lvl = potential_skills[potential_actions.index(action)]
        return action, skill_lvl, potential_actions

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
        embedding = self.qnetwork_local.forward(states_t)
        taus, taus_, entropy = self.fpn(embedding.detach())

        # Quantiles for current state-action
        f_z_expected = self.qnetwork_local.get_quantiles(states_t, taus_, embedding)
        q_expected = f_z_expected.gather(
            2, actions_t.unsqueeze(-1).expand(self.batch_size, self.n_quantiles, 1)
        )
        assert q_expected.shape == (self.batch_size, self.n_quantiles, 1)

        # Fraction loss
        with torch.inference_mode():
            f_z_tau = self.qnetwork_local.get_quantiles(
                states_t, taus[:, 1:-1], embedding.detach()
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
                next_embedding_loc = self.qnetwork_local.forward(next_states_t)
                n_taus, n_taus_, _ = self.fpn(next_embedding_loc)
                f_z_next_loc = self.qnetwork_local.get_quantiles(
                    next_states_t, n_taus_, next_embedding_loc
                )
                q_targets_next_loc = (
                    (n_taus[:, 1:].unsqueeze(-1) - n_taus[:, :-1].unsqueeze(-1))
                    * f_z_next_loc
                ).sum(1)
                action_idx = torch.argmax(q_targets_next_loc, dim=1, keepdim=True)

                next_embedding = self.qnetwork_target.forward(next_states_t)
                f_z_next = self.qnetwork_target.get_quantiles(
                    next_states_t, taus_, next_embedding
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
            ns_embedding = self.qnetwork_target.forward(next_states_t).detach()
            ns_taus, ns_taus_, ns_entropy = self.fpn(ns_embedding.detach())
            ns_taus = ns_taus.detach()
            ns_entropy = ns_entropy.detach()

            m_quantiles = self.qnetwork_target.get_quantiles(
                next_states_t, ns_taus_, ns_embedding
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
                states_t, taus_, embedding
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

        embedding = self.qnetwork_local.forward(states_t)
        taus, taus_, entropy = self.fpn(embedding.detach())

        f_z_expected = self.qnetwork_local.get_quantiles(states_t, taus_, embedding)
        q_expected = f_z_expected.gather(
            2, actions_t.unsqueeze(-1).expand(self.batch_size, self.n_quantiles, 1)
        )

        with torch.inference_mode():
            f_z_tau = self.qnetwork_local.get_quantiles(
                states_t, taus[:, 1:-1], embedding.detach()
            )
            fz_tau = f_z_tau.gather(
                2, actions_t.unsqueeze(-1).expand(self.batch_size, self.n_quantiles - 1, 1)
            )

        frac_loss = calc_fraction_loss(q_expected.detach(), fz_tau, taus, weights=weights_t)
        frac_loss = frac_loss + self.entropy_coeff * entropy.mean()

        # Targets (munchausen or not) - keep original logic
        if not self.munchausen:
            with torch.inference_mode():
                next_embedding_loc = self.qnetwork_local.forward(next_states_t)
                n_taus, n_taus_, _ = self.fpn(next_embedding_loc)
                f_z_next_loc = self.qnetwork_local.get_quantiles(
                    next_states_t, n_taus_, next_embedding_loc
                )
                q_targets_next_loc = (
                    (n_taus[:, 1:].unsqueeze(-1) - n_taus[:, :-1].unsqueeze(-1))
                    * f_z_next_loc
                ).sum(1)
                action_idx = torch.argmax(q_targets_next_loc, dim=1, keepdim=True)

                next_embedding = self.qnetwork_target.forward(next_states_t)
                f_z_next = self.qnetwork_target.get_quantiles(
                    next_states_t, taus_, next_embedding
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
            ns_embedding = self.qnetwork_target.forward(next_states_t).detach()
            ns_taus, ns_taus_, ns_entropy = self.fpn(ns_embedding.detach())
            ns_taus = ns_taus.detach()
            ns_entropy = ns_entropy.detach()

            m_quantiles = self.qnetwork_target.get_quantiles(
                next_states_t, ns_taus_, ns_embedding
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
                states_t, taus_, embedding
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
    """PPO-style actor-critic with action masking and optional curiosity."""

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
        self.am = am

        # PPO-specific LR (kept from your original)
        self.actor_lr = 3e-4
        self.critic_lr = 1e-3

        # Rollout storage: (state, action, reward, next_state, done, invalid_actions)
        self.rollout_storage = []
        self.last_invalid_actions: Optional[List[int]] = None

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

        actor_parameters = (
            list(self.model.actor_body.parameters()) + list(self.model.policy_head.parameters())
        )
        critic_parameters = (
            list(self.model.critic_body.parameters()) + list(self.model.value_head.parameters())
        )

        self.actor_optimizer = optim.Adam(actor_parameters, lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(critic_parameters, lr=self.critic_lr)

    def act(
        self, state: np.ndarray, all_ff_waiting, eps: float = 0.0
    ) -> Tuple[int, int, List[int]]:
        """Sample an action from the masked categorical distribution.

        Returns the feasible actions alongside the choice: callers use them for
        explainability (which alternatives were available and rejected).
        """
        potential_actions, potential_skills = get_potential_actions(state, all_ff_waiting)
        state_t = torch.from_numpy(state).float().to(self.device)

        self.model.eval()
        with torch.inference_mode():
            logits, _value = self.model(state_t)
        self.model.train()

        logits = logits.squeeze(0)
        invalid_actions = [a for a in range(self.action_size) if a not in potential_actions]

        masked_logits = logits.clone()
        if invalid_actions:
            masked_logits[invalid_actions] = -1e9

        dist = torch.distributions.Categorical(logits=masked_logits)
        action = int(dist.sample().item())

        self.last_invalid_actions = invalid_actions
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
        """Store rollout and trigger PPO update once we have a batch."""
        state_t = torch.from_numpy(state).float()
        next_state_t = torch.from_numpy(next_state).float()

        invalid_actions = self.last_invalid_actions or []
        self.rollout_storage.append((state_t, action, reward, next_state_t, done, invalid_actions))
        self.last_invalid_actions = None

        self.t_step += 1
        if self.t_step % self.batch_size == 0:
            return self.learn()

        return None

    def learn(self) -> Optional[Tuple[Tensor, Tensor]]:
        """Update actor and critic on the stored rollout (simple PPO-style update)."""
        if not self.rollout_storage:
            return None

        states = torch.stack([m[0] for m in self.rollout_storage]).to(self.device)
        actions = torch.tensor([m[1] for m in self.rollout_storage], dtype=torch.long, device=self.device)
        rewards = torch.tensor([m[2] for m in self.rollout_storage], dtype=torch.float32, device=self.device)
        next_states = torch.stack([m[3] for m in self.rollout_storage]).to(self.device)
        dones = [m[4] for m in self.rollout_storage]
        invalid_actions = [m[5] for m in self.rollout_storage]

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

        logits, values = self.model(states)

        masked_logits = logits.clone()
        for i, inval in enumerate(invalid_actions):
            if inval:
                masked_logits[i, inval] = -1e9

        dist = torch.distributions.Categorical(logits=masked_logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # Optional Munchausen shaping (note: PPO typically doesn't use this)
        if self.munchausen:
            rewards = rewards + self.alpha * torch.clamp(log_probs.detach(), min=self.lo, max=0.0)

        # Monte-Carlo returns
        returns: List[float] = []
        running_return = 0.0
        for reward, done in zip(reversed(rewards.detach().cpu().tolist()), reversed(dones)):
            if done:
                running_return = 0.0
            running_return = reward + self.gamma * running_return
            returns.insert(0, running_return)

        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device).unsqueeze(1)

        advantages = returns_t - values.detach()
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Actor loss: policy gradient + entropy bonus
        actor_loss = -(log_probs * advantages.squeeze(-1)).mean() - self.entropy_coeff * entropy

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        clip_grad_norm_(
            list(self.model.actor_body.parameters()) + list(self.model.policy_head.parameters()),
            self.grad_clip,
        )
        self.actor_optimizer.step()

        # Critic loss: value regression
        self.critic_optimizer.zero_grad()
        _logits, values = self.model(states)
        critic_loss = F.mse_loss(values.squeeze(-1), returns_t.squeeze(-1))
        critic_loss.backward()
        clip_grad_norm_(
            list(self.model.critic_body.parameters()) + list(self.model.value_head.parameters()),
            self.grad_clip,
        )
        self.critic_optimizer.step()

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
