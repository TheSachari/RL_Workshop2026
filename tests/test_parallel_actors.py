"""Tests for the actor pool's pure parts.

The pool itself spawns processes and loads an environment per actor, which is
too slow for the suite; `tools/` benchmarks cover that. What is tested here is
the logic that decides *what* each actor gets and *how* results come back,
which is where a silent error would corrupt training data rather than crash.
"""

import pytest

from parallel_actors import ActorResult, _slice_bounds, drain, merge_metrics


class TestSliceBounds:
    def test_slices_cover_the_stream_exactly(self):
        bounds = _slice_bounds(100, 4)
        assert bounds[0][0] == 0
        assert bounds[-1][1] == 100
        for (_, end), (start, _) in zip(bounds, bounds[1:]):
            assert end == start          # contiguous, no gap or overlap

    def test_slices_are_contiguous_not_strided(self):
        """State carries across events -- a vehicle dispatched at one event
        returns at a later one -- so an actor needs an unbroken window."""
        bounds = _slice_bounds(10, 2)
        assert bounds == [(0, 5), (5, 10)]

    def test_uneven_splits_lose_no_events(self):
        bounds = _slice_bounds(10, 3)
        assert sum(end - start for start, end in bounds) == 10

    def test_more_actors_than_events_drops_the_empty_ones(self):
        bounds = _slice_bounds(3, 8)
        assert all(end > start for start, end in bounds)
        assert sum(end - start for start, end in bounds) == 3

    def test_zero_actors_is_an_error(self):
        with pytest.raises(ValueError):
            _slice_bounds(100, 0)


def result(actor_id=0, transitions=(), **metrics):
    return ActorResult(actor_id=actor_id, transitions=list(transitions),
                       metrics=metrics, events_consumed=0)


class TestMergeMetrics:
    def test_counters_are_summed(self):
        merged = merge_metrics([result(0, v_sent=3), result(1, v_sent=4)])
        assert merged["v_sent"] == 7

    def test_reserve_levels_are_averaged_not_summed(self):
        """`_disp` is a state -- vehicles currently in station -- not a tally,
        so adding two actors' values would report an impossible fleet."""
        merged = merge_metrics([result(0, VSAV_disp=2), result(1, VSAV_disp=4)])
        assert merged["VSAV_disp"] == 3

    def test_empty_input(self):
        assert merge_metrics([]) == {}


class _RecordingAgent:
    """Counts what the learner is asked to do."""

    gamma = 0.99

    def __init__(self, loss_every=None):
        self.observed = []
        self.train_calls = 0
        self._loss_every = loss_every

    def observe(self, *transition):
        self.observed.append(transition)

    def train_step(self):
        self.train_calls += 1
        if self._loss_every and self.train_calls % self._loss_every == 0:
            return 0.5
        return None


class TestDrain:
    def test_every_transition_reaches_the_learner(self):
        agent = _RecordingAgent()
        results = [
            result(0, [("s0", 1, 1.0, "s1", False)]),
            result(1, [("s2", 2, 2.0, "s3", True)]),
        ]

        drain(agent, results)

        assert agent.observed == [
            ("s0", 1, 1.0, "s1", False),
            ("s2", 2, 2.0, "s3", True),
        ]

    def test_only_real_losses_are_returned(self):
        """`train_step` returns None when no batch was due; those must not be
        recorded as zero-loss updates."""
        agent = _RecordingAgent(loss_every=2)
        results = [result(0, [("s", 0, 0.0, "s", False)] * 4)]

        losses = drain(agent, results)

        assert losses == [0.5, 0.5]
        assert agent.train_calls == 4

    def test_eval_mode_stores_without_learning(self):
        agent = _RecordingAgent()
        drain(agent, [result(0, [("s", 0, 0.0, "s", False)])], train=False)
        assert agent.observed and agent.train_calls == 0
