"""The agent interface: choose actions and, for learning agents, define a loss."""

from __future__ import annotations

import dataclasses
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, PyTree

from bomberman.game.const import ACTION
from bomberman.game.observe import Observation

N_ACTIONS = len(ACTION)


def _configure(obj: Any, changes: dict[str, Any]) -> Any:
    """Copy ``obj`` with some fields replaced, without calling ``__init__``."""
    fields = {f.name for f in dataclasses.fields(obj)}
    unknown = set(changes) - fields
    if unknown:
        raise TypeError(f"{type(obj).__name__} has no field {sorted(unknown)}")

    clone = object.__new__(type(obj))
    for name in fields:
        object.__setattr__(clone, name, changes.get(name, getattr(obj, name)))
    return clone


class Batch(eqx.Module):
    """The part of a rollout that belongs to one agent, as passed to ``Agent.loss``.

    Fields have shape ``(games, rounds, ...)``. ``mask`` is False for ticks after the game ended.
    """

    obs: Observation      # (B, R) what the agent saw
    legal_mask: Array     # (B, R, N_ACTIONS) bool
    action: Array         # (B, R) int32, the action it took
    reward: Array         # (B, R) float32, its reward
    mask: Array           # (B, R) bool, False once the game is over
    key: PRNGKeyArray     # (B, R) random key passed to `act`, to repeat the same sampling
    progress: Array       # training progress, 0 on the first update and 1 on the last
    opt_state: Any        # the agent's optimiser state, for learning rate schedules

    @property
    def trajectories(self) -> int:
        """Number of games in the batch."""
        return self.action.shape[0]

    @property
    def rounds(self) -> int:
        return self.action.shape[1]


class Agent[AgentState: PyTree](eqx.Module):
    """Controls one player for a whole game.

    ``AgentState`` is whatever the agent remembers during a game, or ``None`` if it remembers nothing.
    """

    def init(self) -> AgentState:
        """The state passed to the first ``act`` call of a game."""
        return None

    def act(
        self,
        obs: Observation,
        legal_mask: Array,
        key: PRNGKeyArray,
        state: AgentState = None,
    ) -> tuple[Array, Array | None, AgentState]:
        """Choose an action and return ``(action, value, state)``.

        ``value`` may be ``None``. The method must work under ``jit`` and ``vmap``.
        """
        raise NotImplementedError

    def greedy_action(
        self,
        obs: Observation,
        legal_mask: Array,
        key: PRNGKeyArray,
        state: AgentState = None,
    ) -> tuple[Array, Array | None, AgentState]:
        """The agent's best action, used for evaluation. ``key`` only breaks ties."""
        return self.act(obs, legal_mask, key, state)

    def loss(self, batch: Batch) -> tuple[Array, dict[str, Array]]:
        """Loss and metrics for a batch. By default the agent does not learn."""
        return jnp.float32(0.0), {}

    def configure(self, **changes) -> Agent:
        """A copy with some fields replaced. The weights stay the same."""
        return _configure(self, changes)

