"""Resumable training checkpoints.

Saving `qnetwork_local.state_dict()` alone is not enough to resume a run: the
weights are only one of the things a training step reads. Everything else --
where each firefighter is, which vehicles are out, the cumulative indicators the
reward is a delta of, epsilon and its decay counter, the optimiser moments, the
replay buffer, the RNG streams -- was rebuilt from scratch on restart, so a
"resumed" run silently restarted exploration at eps=1.0 against a fresh
environment while claiming to continue.

This module snapshots all of it, so `--resume` continues the run rather than
starting a new one that happens to share weights.

What is captured
----------------
* **Position** -- the event index reached, so the stream restarts after it.
* **Environment** -- `dic_ff`, `dic_vehicles`, `dic_inter`, `planning`,
  `dic_lent`, plus `dic_indic`/`dic_indic_old`. The indicator pair matters more
  than it looks: `compute_reward` reads the *difference* between them, so
  dropping them makes the first reward after a resume meaningless.
* **Loop state** -- the `_LoopState` fields that outlive a single event
  (`current_ff_inter`, `dic_log`, `dic_back`, `dic_start_time`, `dic_veh_typ`,
  `vehicle_out`), the `Fleet` reinforcement bookkeeping, and the two dates the
  loop carries across rows (`old_date`, `date_reference`).
* **Agent** -- both networks (target included; it was never saved before, so a
  resume used a randomly initialised target and produced garbage TD targets),
  the optimiser(s), `t_step` (which drives the LR schedule and the
  `update_every` cadence), and FQF's `fpn`/`frac_optimizer` and ICM when built.
* **Replay buffer** -- transitions, priorities and the SumTree, so learning does
  not restart against an empty memory.
* **RNG** -- `random`, `numpy` and `torch` (CPU and CUDA), because `act()` draws
  from `np.random`/`random` and `sample()` draws from the buffer's stream.

What is deliberately *not* claimed
----------------------------------
A resumed run is statistically equivalent to an uninterrupted one, not
bit-identical to it. Two reasons, both structural: the event stream is resumed
by skipping to an index rather than replaying `itertuples` state, and cuDNN
kernels are not deterministic by default. Treat a resume as a faithful
continuation, not as a reproduction.

Writes are atomic (temp file + `os.replace`) and kept on a rotation of two, so a
process killed mid-save leaves the previous checkpoint intact -- which is
exactly how the last run ended.
"""

from __future__ import annotations

import os
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Bumped when the payload layout changes in a way older files cannot satisfy.
CKPT_VERSION = 1

# Fields of `_LoopState` that carry meaning across events. The per-row and
# per-intervention scratch fields are excluded on purpose: `_unpack_row` and
# `_handle_intervention` overwrite them before they are read, so persisting them
# would store values that are never consulted.
LOOP_FIELDS = (
    "vehicle_out",
    "action_num",
    "current_ff_inter",
    "dic_log",
    "dic_back",
    "dic_start_time",
    "dic_veh_typ",
)

# Environment containers the simulation mutates in place.
ENV_FIELDS = (
    "dic_vehicles",
    "dic_inter",
    "dic_ff",
    "dic_indic",
    "dic_indic_old",
    "planning",
    "dic_lent",
)


def _rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # The generator state must live on the CPU regardless of the training device.
    torch.set_rng_state(torch.as_tensor(state["torch"], dtype=torch.uint8).cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        # Only restore if the GPU count matches; otherwise the per-device states
        # do not line up and torch raises deep inside the CUDA allocator.
        if len(state["cuda"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(
                [torch.as_tensor(s, dtype=torch.uint8).cpu() for s in state["cuda"]]
            )


def _agent_state(agent: Any, *, include_buffer: bool) -> dict:
    """Every mutable piece of the agent, not just the policy weights."""
    state: dict = {
        "qnetwork_local": agent.qnetwork_local.state_dict(),
        "t_step": agent.t_step,
    }

    # Present on every agent variant used here, but guarded so a variant that
    # drops one does not break checkpointing for the others.
    if hasattr(agent, "qnetwork_target"):
        state["qnetwork_target"] = agent.qnetwork_target.state_dict()
    if hasattr(agent, "optimizer"):
        state["optimizer"] = agent.optimizer.state_dict()

    # FQF-only: the fraction proposal network and its own RMSprop optimiser.
    if getattr(agent, "fpn", None) is not None:
        state["fpn"] = agent.fpn.state_dict()
    if getattr(agent, "frac_optimizer", None) is not None:
        state["frac_optimizer"] = agent.frac_optimizer.state_dict()

    # Curiosity module, built only when `curiosity != 0`.
    icm = getattr(agent, "icm", None)
    if icm is not None:
        state["icm"] = icm.state_dict()
        if hasattr(icm, "optimizer"):
            state["icm_optimizer"] = icm.optimizer.state_dict()

    if include_buffer:
        state["memory"] = _buffer_state(agent.memory)

    return state


def _plain(experience: Any) -> tuple:
    """Reduce a buffer entry to a plain tuple of CPU tensors.

    Two problems solved here. The buffers build their `Experience` namedtuple
    inside `__init__`, so the class is not reachable at module level and pickle
    cannot resolve it -- plain tuples sidestep that, and `_rebuild` restores the
    namedtuple from the live buffer's class.

    The `.cpu()` matters just as much: `sample()` calls `np.stack` on the stored
    states, which fails on CUDA tensors. Transitions are added as CPU tensors, so
    forcing CPU here keeps them that way even when the checkpoint is loaded with
    `map_location="cuda"`.
    """
    return tuple(
        v.detach().cpu() if isinstance(v, torch.Tensor) else v for v in experience
    )


def _rebuild(memory: Any, entries: list) -> list:
    """Turn stored tuples back into the live buffer's Experience type."""
    factory = getattr(memory, "experience", None)
    entries = [_plain(e) for e in entries]
    if factory is None:
        return entries
    return [factory(*e) for e in entries]


def _buffer_state(memory: Any) -> dict:
    """Snapshot a replay buffer across the three implementations in use.

    The buffers differ in what backs them (`deque`, a priorities array, a
    SumTree), so copy whichever attributes are present rather than assuming one
    layout. `n_step_buffer` is included because a transition only enters memory
    once the n-step window closes -- dropping it loses the partial window.
    """
    state: dict = {"memory": [_plain(e) for e in memory.memory]}

    for attr in ("pos", "frame", "beta", "current_size"):
        if hasattr(memory, attr):
            state[attr] = getattr(memory, attr)

    if hasattr(memory, "n_step_buffer"):
        state["n_step_buffer"] = [_plain(e) for e in memory.n_step_buffer]
    if hasattr(memory, "priorities"):
        state["priorities"] = np.array(memory.priorities, copy=True)
    if getattr(memory, "sum_tree", None) is not None:
        # SumTree stores its payload in plain arrays; copy them out directly.
        tree = memory.sum_tree
        state["sum_tree"] = {
            k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v)
            for k, v in vars(tree).items()
        }

    return state


def _restore_buffer(memory: Any, state: dict) -> None:
    memory.memory.clear()
    memory.memory.extend(_rebuild(memory, state["memory"]))

    for attr in ("pos", "frame", "beta", "current_size"):
        if attr in state and hasattr(memory, attr):
            setattr(memory, attr, state[attr])

    if "n_step_buffer" in state and hasattr(memory, "n_step_buffer"):
        memory.n_step_buffer.clear()
        memory.n_step_buffer.extend(_rebuild(memory, state["n_step_buffer"]))

    if "priorities" in state and hasattr(memory, "priorities"):
        memory.priorities = state["priorities"]

    if "sum_tree" in state and getattr(memory, "sum_tree", None) is not None:
        for k, v in state["sum_tree"].items():
            setattr(memory.sum_tree, k, v)


def _restore_agent(agent: Any, state: dict) -> None:
    agent.qnetwork_local.load_state_dict(state["qnetwork_local"])
    agent.t_step = state["t_step"]

    if "qnetwork_target" in state and hasattr(agent, "qnetwork_target"):
        agent.qnetwork_target.load_state_dict(state["qnetwork_target"])
    if "optimizer" in state and hasattr(agent, "optimizer"):
        agent.optimizer.load_state_dict(state["optimizer"])

    if "fpn" in state and getattr(agent, "fpn", None) is not None:
        agent.fpn.load_state_dict(state["fpn"])
    if "frac_optimizer" in state and getattr(agent, "frac_optimizer", None) is not None:
        agent.frac_optimizer.load_state_dict(state["frac_optimizer"])

    icm = getattr(agent, "icm", None)
    if icm is not None and "icm" in state:
        icm.load_state_dict(state["icm"])
        if "icm_optimizer" in state and hasattr(icm, "optimizer"):
            icm.optimizer.load_state_dict(state["icm_optimizer"])

    if "memory" in state:
        _restore_buffer(agent.memory, state["memory"])


def save(
    path: Path,
    *,
    agent: Any,
    env: Any,
    fleet: Any,
    loop: Any,
    rl: dict,
    num_inter: int,
    row_index: Any,
    old_date: Any,
    date_reference: Any,
    reward_evo: list,
    dic_saved_skills: dict,
    args: Any,
    include_buffer: bool = True,
    keep: int = 2,
) -> Path:
    """Write a resumable checkpoint atomically, keeping the last `keep` files.

    The temp-file-then-`os.replace` dance is the point: `os.replace` is atomic
    on POSIX, so a kill during the (multi-second, buffer-sized) write leaves the
    previous checkpoint whole instead of truncating the live one.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": CKPT_VERSION,
        "num_inter": num_inter,
        "row_index": row_index,
        "agent": _agent_state(agent, include_buffer=include_buffer),
        "env": {f: getattr(env, f) for f in ENV_FIELDS},
        "skills_updated": env.skills_updated,
        "old_date": old_date,
        "date_reference": date_reference,
        "fleet": fleet,
        "loop": {f: getattr(loop, f) for f in LOOP_FIELDS} if loop is not None else {},
        "rl": {k: rl[k] for k in ("eps", "d", "score", "action_num") if k in rl},
        "reward_evo": reward_evo,
        "dic_saved_skills": dic_saved_skills,
        # Recorded so a resume can refuse a checkpoint from a different setup
        # rather than silently continuing with mismatched settings.
        "run": {
            "dataset": args.dataset,
            "start": args.start,
            "end": args.end,
            "agent_model": args.agent_model,
            "n_hours": args.n_hours,
            "top_n": args.top_n,
            "constraint_factor_veh": args.constraint_factor_veh,
            "constraint_factor_ff": args.constraint_factor_ff,
            "seed": args.seed,
        },
        "rng": _rng_state(),
    }

    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(payload, f, pickle_protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())

    # Rotate *before* the replace: the live file still holds the previous
    # checkpoint at this point, which is exactly what `.prev` should capture.
    # Rotating afterwards would link `.prev` to the inode the new file now
    # occupies, leaving two names for one version and no fallback at all.
    _rotate(path, keep)
    os.replace(tmp, path)

    return path


def _rotate(path: Path, keep: int) -> None:
    """Point `.prev` at the checkpoint currently live, before it is replaced."""
    if keep < 2 or not path.exists():
        return
    prev = path.with_suffix(path.suffix + ".prev")
    try:
        if prev.exists():
            prev.unlink()
        # Hard link rather than copy: same bytes, no second write of a
        # buffer-sized file, and unlinking one does not touch the other.
        os.link(path, prev)
    except OSError:
        # A rotation failure must never take the run down; the live checkpoint
        # is already safely written by this point.
        pass


def load(path: Path, map_location: str = "cpu") -> dict:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    version = payload.get("version")
    if version != CKPT_VERSION:
        raise ValueError(
            f"Checkpoint {path} has version {version!r}, expected {CKPT_VERSION}. "
            "It was written by a different version of checkpoint.py."
        )
    return payload


def check_compatible(payload: dict, args: Any) -> None:
    """Refuse a checkpoint whose run settings differ from the current ones.

    Resuming a 10-year run from a 2-year checkpoint, or with a different reward
    dataset, produces a run that looks fine and means nothing. `end` is compared
    because `eps_update` and `max_train_steps` are derived from `end - start`,
    so changing it silently rescales the exploration schedule.
    """
    run = payload.get("run", {})
    current = {
        "dataset": args.dataset,
        "start": args.start,
        "end": args.end,
        "agent_model": args.agent_model,
        "n_hours": args.n_hours,
        "top_n": args.top_n,
        "constraint_factor_veh": args.constraint_factor_veh,
        "constraint_factor_ff": args.constraint_factor_ff,
        "seed": args.seed,
    }
    mismatched = {k: (run.get(k), v) for k, v in current.items() if run.get(k) != v}
    if mismatched:
        detail = ", ".join(
            f"{k}: checkpoint={old!r} current={new!r}"
            for k, (old, new) in sorted(mismatched.items())
        )
        raise ValueError(
            "Checkpoint was written by a run with different settings "
            f"({detail}). Resuming would not continue that run."
        )


def restore(
    payload: dict,
    *,
    agent: Any,
    env: Any,
    rl: dict,
) -> dict:
    """Apply a checkpoint to a freshly built agent and environment.

    Returns the pieces the caller must thread into the simulation loop itself
    (position, fleet, loop fields, dates), which cannot be set from here because
    `_LoopState` is built inside `run_simulation`.
    """
    _restore_agent(agent, payload["agent"])

    for f in ENV_FIELDS:
        setattr(env, f, payload["env"][f])
    env.skills_updated = payload["skills_updated"]
    env.old_date = payload["old_date"]
    env.date_reference = payload["date_reference"]

    rl.update(payload["rl"])

    _restore_rng(payload["rng"])

    return {
        "num_inter": payload["num_inter"],
        "row_index": payload["row_index"],
        "fleet": payload["fleet"],
        "loop": payload["loop"],
        "reward_evo": payload["reward_evo"],
        "dic_saved_skills": payload["dic_saved_skills"],
    }
