"""Run several environment actors at once, feeding one learner.

The simulation is bound by a single CPU core: the loop is sequential Python and
the GPU sits at about 16% during training, because `agent.act` does one
batch-of-one forward per decision and `agent.step` then learns synchronously in
the middle of the rollout.

Batching the decisions *within* a rollout is not available. The roles of a
vehicle are filled one at a time and each `step` mutates `planning`, so the
next decision's candidate set depends on the previous choice -- there is
nothing to batch. The parallelism has to come from running several
environments.

That was impractical while forking an environment meant a 2.7 s deepcopy of
the planning. `Environment.fork` is now 38 ms, so an actor per core is cheap.

Design
------
Actors are separate *processes*, not threads: the loop is pure Python and the
GIL would serialise them. Each gets

* its own `Environment`, forked from one load, so the read-only tables
  (event stream, skills frame, role and distance tables) are shared by the OS
  through copy-on-write rather than duplicated;
* its own slice of the event stream, so no two actors replay the same events;
* a frozen copy of the policy weights, refreshed between rounds.

They send transitions back over a queue. The parent owns the network, calls
`agent.observe` for each transition and `agent.train_step` on its own
schedule -- the split that `step` used to fuse.

What this does not do
---------------------
It does not make one run faster, and **the slices do not reconstruct a full
run**. Each actor starts from a freshly forked environment, so an actor holding
the second half of a stream begins with every vehicle in station and every crew
rested, where a continuous run would have arrived there mid-dispatch. Summing
two halves of an 800-intervention stream against one continuous run moves
`v_degraded` by +3% and `v3_not_sent_from_s3` by +2%, with most other counters
inside 1%.

So this is for *collecting training transitions*, where the samples only have
to be representative, and for replicate sweeps, where the runs are meant to be
independent anyway. Reported metrics must still come from a single continuous
`run_simulation`; `merge_metrics` exists to monitor a training round, not to
produce a result.

The actors are also off-policy by construction: an actor collects a round with
the weights it was handed, so its transitions are up to one round stale by the
time they are learned from. That is the same staleness Ape-X accepts, and the
replay buffer already assumes off-policy data.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Sequence

import numpy as np


@dataclass
class ActorResult:
    """What one actor sends back after a round."""

    actor_id: int
    transitions: List[tuple]        # (state, action, reward, next_state, done)
    metrics: dict
    events_consumed: int


def _slice_bounds(total: int, n: int) -> List[tuple]:
    """Split `total` events into `n` contiguous, near-equal ranges.

    Contiguous rather than strided: the simulation carries state across events
    (a vehicle dispatched at one event returns at a later one), so an actor
    needs an unbroken window to stay coherent.
    """
    if n <= 0:
        raise ValueError("need at least one actor")
    edges = np.linspace(0, total, n + 1).astype(int)
    return [(int(a), int(b)) for a, b in zip(edges, edges[1:]) if b > a]


def _actor_main(actor_id, env_factory, weights, eps, bounds, out_queue):
    """One actor process: build an environment, run its slice, ship transitions.

    `env_factory` is called in the child rather than the environment being
    pickled across, because the event stream and skills frame are large and the
    fork already shares them through the OS.
    """
    try:
        env, fleet, agent = env_factory()
        if weights is not None:
            agent.qnetwork_local.load_state_dict(weights)

        collected: List[tuple] = []

        from simulator import run_simulation
        from transition import PendingTransition

        pending = PendingTransition(gamma=agent.gamma)

        def decide(state, all_ff_waiting, ff_array, inter_done):
            transition = pending.close(state, inter_done)
            if transition is not None:
                collected.append((
                    transition.state, transition.action, transition.reward,
                    transition.next_state, transition.done,
                ))
            action, skill_lvl, potential_actions = agent.act(
                state, all_ff_waiting, eps
            )
            pending.open(state, action)
            return action, skill_lvl, potential_actions

        def on_action(ctx):
            pending.add_reward(env_factory.reward_fn(ctx))

        start, end = bounds
        env.df_pc = env.df_pc.iloc[start:end]

        run_simulation(env, fleet, decide, action_size=env_factory.action_size,
                       on_action=on_action)

        out_queue.put(ActorResult(
            actor_id=actor_id,
            transitions=collected,
            metrics=dict(env.dic_indic),
            events_consumed=end - start,
        ))
    except Exception as exc:                      # noqa: BLE001
        # A crashed actor must not hang the parent on `queue.get`.
        out_queue.put(exc)


class ActorPool:
    """A pool of environment actors feeding one learner.

    Parameters
    ----------
    env_factory : callable
        Called in each child; returns `(env, fleet, agent)`. Must also carry
        `action_size` and `reward_fn` attributes, which the actor needs and
        which are cheaper to attach than to pickle separately.
    n_actors : int
        Defaults to 4, which is where this measures out -- not to the core
        count. Each actor loads its own environment, which costs about 1.2 s
        of pandas and pickle work that no amount of forking avoids, so the
        speedup peaks and then falls:

            4000 interventions, 12-core machine, heuristic policy
            1 process   6.58 s   1.00x
            4 processes 3.29 s   2.00x
            8 processes 4.29 s   1.54x
            12 processes 6.81 s  0.97x

        The same sweep over 800 interventions peaks at 1.23x with two actors:
        the fixed load cost only amortises once each actor has real work. Below
        a few thousand events per actor, run it sequentially.
    """

    #: Measured optimum; see the class docstring for the sweep behind it.
    DEFAULT_ACTORS = 4

    def __init__(self, env_factory: Callable, n_actors: Optional[int] = None):
        self.env_factory = env_factory
        if n_actors is None:
            n_actors = min(self.DEFAULT_ACTORS, os.cpu_count() or 1)
        self.n_actors = n_actors
        self._ctx = mp.get_context("spawn")

    def collect(self, weights, eps: float, total_events: int) -> List[ActorResult]:
        """Run one round: every actor plays its slice, then all are joined.

        Returns results in actor order regardless of completion order, so a
        round is reproducible given the same weights and seeds.
        """
        bounds = _slice_bounds(total_events, self.n_actors)
        queue = self._ctx.Queue()
        procs = []

        for actor_id, window in enumerate(bounds):
            p = self._ctx.Process(
                target=_actor_main,
                args=(actor_id, self.env_factory, weights, eps, window, queue),
            )
            p.start()
            procs.append(p)

        results: List[Any] = []
        for _ in procs:
            item = queue.get()
            if isinstance(item, Exception):
                for p in procs:
                    p.terminate()
                raise item
            results.append(item)

        for p in procs:
            p.join()

        results.sort(key=lambda r: r.actor_id)
        return results


def drain(agent, results: Iterable[ActorResult], train: bool = True) -> List[float]:
    """Feed a round's transitions to the learner.

    `observe` for every transition, `train_step` after each -- the agent's own
    `update_every` decides which of those actually run a batch, exactly as it
    did when `step` fused the two.
    """
    losses = []
    for result in results:
        for state, action, reward, next_state, done in result.transitions:
            agent.observe(state, action, reward, next_state, done)
            if train:
                loss = agent.train_step()
                if loss is not None:
                    losses.append(loss)
    return losses


def merge_metrics(results: Sequence[ActorResult]) -> dict:
    """Sum the per-actor indicator counters.

    Sound because every counter in `dic_indic` is a count of events in a
    disjoint slice of the stream. The three `_disp` reserve *levels* are the
    exception -- they are a state, not a tally -- so they are averaged.
    """
    if not results:
        return {}

    levels = {"VSAV_disp", "FPT_disp", "EPA_disp"}
    merged: dict = {}
    for result in results:
        for key, value in result.metrics.items():
            merged[key] = merged.get(key, 0) + value
    for key in levels & merged.keys():
        merged[key] /= len(results)
    return merged
