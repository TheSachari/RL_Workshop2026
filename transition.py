"""The one-step lag between choosing an action and being able to learn from it.

A transition needs `(s, a, r, s', done)`, but the simulator hands those out at
two different moments: `decide` sees `s` and picks `a`, then the environment
mutates and `on_action` can finally compute `r` from the indicator deltas. The
next state `s'` is not known until the *following* `decide`. So one decision's
transition is only complete one decision later.

That lag used to live as five keys in a shared `rl` dict -- `old_state`,
`action`, `reward`, `compute`, `potential_old` -- written in one callback and
read in the other, with `compute` flagging "the first decision has no
predecessor". The fields were mutable, reachable from every callback, and
nothing tied them together: the invariant that `old_state` and `action` belong
to the *same* decision was maintained by the two assignments at the bottom of
`decide` happening to run every time.

`PendingTransition` makes that lag the whole of one object's job. `open` records
the half a decision knows; `close` supplies the next state and returns a
complete transition or None on the first call. Nothing else can see a
half-built one, and the fields cannot drift apart because they are only ever
written together.

Potential-based shaping rides along here for the same reason: the term needs
the potential of both states, and this is the only place that holds both.
"""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class Transition:
    """A complete `(s, a, r, s', done)`, ready for `agent.step`."""

    state: Any
    action: int
    reward: float
    next_state: Any
    done: bool


class PendingTransition:
    """Holds the half-decision until the next state arrives.

    Parameters
    ----------
    gamma : float
        Discount used by the shaping term. Must match the agent's, or the
        shaping stops telescoping and leaves a residual bias.
    shaping_coeff : float
        Weight on `gamma * potential' - potential`. Zero disables shaping, and
        then potentials are neither read nor required.
    """

    __slots__ = ("gamma", "shaping_coeff", "_state", "_action", "_reward",
                 "_potential", "_open")

    def __init__(self, gamma: float, shaping_coeff: float = 0.0) -> None:
        self.gamma = gamma
        self.shaping_coeff = shaping_coeff
        self._open = False
        self._state = None
        self._action = -1
        self._reward = 0.0
        self._potential = None

    @property
    def is_open(self) -> bool:
        """True once a decision is waiting for its next state."""
        return self._open

    def open(self, state, action: int, potential: Optional[float] = None) -> None:
        """Record the half of a transition that `decide` knows."""
        self._state = state
        self._action = action
        self._potential = potential
        self._reward = 0.0
        self._open = True

    def add_reward(self, reward: float) -> None:
        """Accumulate the environment reward for the decision in flight.

        Additive rather than assigning: a decision that draws more than one
        reward callback should sum them, and the previous code's plain
        assignment would have kept only the last.
        """
        if self._open:
            self._reward += reward

    def close(self, next_state, done: bool,
              potential: Optional[float] = None) -> Optional[Transition]:
        """Complete the pending transition against `next_state`.

        Returns None when nothing is pending -- the first decision of a run, or
        the first after a resume -- which is what the old `compute` flag
        guarded. The pending slot is cleared either way, so a transition can
        never be emitted twice.
        """
        if not self._open:
            return None

        reward = self._reward
        # Shaping is applied here rather than where the environment reward is
        # computed: the term needs the potential of the *next* state, and this
        # is the first point at which both are known.
        if self.shaping_coeff and potential is not None and self._potential is not None:
            reward += self.shaping_coeff * (self.gamma * potential - self._potential)

        transition = Transition(
            state=self._state, action=self._action, reward=reward,
            next_state=next_state, done=done,
        )
        self._open = False
        self._state = None
        return transition
