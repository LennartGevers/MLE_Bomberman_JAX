"""What an agent sees of the game.

An observation is the full game state minus two things an agent cannot know: the turn order
of the current tick and where the coins under crates are.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from bomberman.game.environment import EnvironmentState

if TYPE_CHECKING:
    from bomberman.game.game import GameState


class Observation(eqx.Module):
    """The game as one agent sees it.

    Unlike ``GameState`` it has no turn order and no random key, and coins under crates are hidden.
    """

    board: EnvironmentState  # hidden coins are moved to HIDDEN_COIN
    agent_id: Array          # () int32, the agent this observation belongs to
    bomb_cooldown: Array     # (num_agents,) ticks until each seat may plant again
    alive: Array             # (num_agents,) bool
    score: Array             # (num_agents,) float32, game score so far
    step_count: Array        # () int32, ticks played so far

    @property
    def num_agents(self) -> int:
        return self.alive.shape[0]

    @property
    def position(self) -> Array:
        """The position of the observing agent."""
        return self.board.agent_positions[self.agent_id]

    @property
    def cooldown(self) -> Array:
        """Ticks until the observing agent may plant a bomb again."""
        return self.bomb_cooldown[self.agent_id]


def observe(state: GameState, agent_id: Array) -> Observation:
    """The part of ``state`` that agent ``agent_id`` is allowed to see."""
    board = state.board
    return Observation(
        board=dataclasses.replace(
            board, coins=board.coins.redacted(board.field_map)
        ),
        agent_id=jnp.asarray(agent_id, jnp.int32),
        bomb_cooldown=state.bomb_cooldown,
        alive=state.alive,
        score=state.score,
        step_count=state.step_count,
    )
