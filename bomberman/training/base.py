"""The interface of reward functions."""

from __future__ import annotations

import equinox as eqx
from jaxtyping import Array

from bomberman.game import GameState


class RewardFn(eqx.Module):
    """Gives every agent a reward for one tick, shape ``(num_agents,)``.

    Rewards can be added (``a + b``) and scaled (``a * 2.0``).
    """

    def __call__(
        self, prev_state: GameState, actions: Array, next_state: GameState
    ) -> Array:
        raise NotImplementedError

    def __add__(self, other: RewardFn) -> RewardFn:
        if not isinstance(other, RewardFn):
            return NotImplemented
        return _SumReward(self, other)

    def __mul__(self, scale: float) -> RewardFn:
        """This reward multiplied by a constant."""
        if not isinstance(scale, (int, float)):
            return NotImplemented
        return _ScaledReward(self, float(scale))

    __rmul__ = __mul__


class _SumReward(RewardFn):
    """Sum of two reward functions."""

    left: RewardFn
    right: RewardFn

    def __call__(self, prev_state, actions, next_state):
        return self.left(prev_state, actions, next_state) + self.right(
            prev_state, actions, next_state
        )


class _ScaledReward(RewardFn):
    """A reward function scaled by a constant."""

    base: RewardFn
    scale: float

    def __call__(self, prev_state, actions, next_state):
        return self.scale * self.base(prev_state, actions, next_state)
