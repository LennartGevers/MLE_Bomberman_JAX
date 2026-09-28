"""A simpler scripted agent, derived from the rule-based agent of the original game."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from bomberman.agents.features import ChannelStack, FeatureShaping
from bomberman.game import ACTION, FIELD, Observation
from bomberman.game.const import DIRECTIONS

from ..base import Agent
from .base import (
    AGENTS,
    BOMBS,
    COINS,
    FIELD_CH,
    FIRE,
    MOVES,
    at,
    bomb_cross_timer,
    free_neighbour_count,
    self_position,
)
from .search import STAY, bfs_first_step


def _touching_crate(field: Array, x: Array, y: Array) -> Array:
    hit = jnp.bool_(False)
    for dx, dy in DIRECTIONS.values():
        hit = hit | (at(field, x + dx, y + dy) == FIELD.CRATE)
    return hit


def _bomb_escape(bombs: Array, x: Array, y: Array, power: int) -> tuple[Array, Array]:
    """Action weights for running away from a bomb in the same row or column."""
    bomb = bombs > 0
    escape = jnp.zeros(6, jnp.float32)
    corner = jnp.zeros(6, jnp.float32)

    # A bomb on our own tile has no "away" direction, so every direction is equally good.
    here = at(bomb, x, y).astype(jnp.float32)
    for move in DIRECTIONS:
        corner = corner.at[move].add(here)

    for r in range(1, power + 1):
        below = at(bomb, x, y + r).astype(jnp.float32)  # same column, larger y
        above = at(bomb, x, y - r).astype(jnp.float32)
        right = at(bomb, x + r, y).astype(jnp.float32)  # same row, larger x
        left = at(bomb, x - r, y).astype(jnp.float32)

        escape = escape.at[ACTION.UP].add(below)
        escape = escape.at[ACTION.DOWN].add(above)
        corner = corner.at[ACTION.LEFT].add(below + above)
        corner = corner.at[ACTION.RIGHT].add(below + above)

        escape = escape.at[ACTION.LEFT].add(right)
        escape = escape.at[ACTION.RIGHT].add(left)
        corner = corner.at[ACTION.UP].add(right + left)
        corner = corner.at[ACTION.DOWN].add(right + left)

    return escape, corner


class CoinCollectorAgent(Agent):
    """Walks to coins, crates and dead ends, bombs crates and runs away from explosions."""

    power: int = eqx.field(static=True, default=3)
    features: FeatureShaping = ChannelStack()

    def act(
        self, obs: Observation, legal_mask: Array, key: PRNGKeyArray, state: None = None
    ) -> tuple[Array, None, None]:
        power = self.power
        planes = self.features(obs)
        field = planes[FIELD_CH]
        agents = planes[AGENTS]
        fire = planes[FIRE]
        bombs = planes[BOMBS]

        pos = self_position(agents)
        x, y = pos[0], pos[1]

        free = field == FIELD.EMPTY
        others = agents > 0
        bomb_grid = bombs > 0
        cross_timer = bomb_cross_timer(bombs, power)

        # Avoid danger. The legal mask already rules out walls, bombs and occupied
        # tiles, but not tiles that are about to explode.
        offs = jnp.asarray([*DIRECTIONS.values(), (0, 0)], jnp.int32)  # moves, WAIT
        cx, cy = x + offs[:, 0], y + offs[:, 1]
        safe = (at(fire, cx, cy) < 1) & (at(cross_timer, cx, cy) > 0)
        valid = legal_mask & (
            jnp.ones(6, jnp.bool_)
            .at[MOVES].set(safe[:-1])
            .at[ACTION.WAIT].set(safe[-1])
        )

        # Targets: coins, crates and dead ends, but no tiles with bombs.
        coins = planes[COINS] > 0
        crates = field == FIELD.CRATE
        dead_ends = free & (free_neighbour_count(free) == 1)
        targets = (coins | crates | dead_ends) & ~bomb_grid
        has_target = targets.any()

        move = bfs_first_step(free & ~others, pos, targets)
        is_move = move != STAY

        # Action priorities. In the original agent later entries win, so we add them up.
        k_base, k_panic = jax.random.split(key)
        pri = (
            jnp.zeros(6, jnp.float32)
            .at[MOVES]
            .add(jax.random.uniform(k_base, MOVES.shape))
        )

        step_action = jnp.where(is_move, move, jnp.int32(ACTION.WAIT))
        pri = pri.at[step_action].add(jnp.where(is_move, 10.0, 0.0))
        pri = pri.at[ACTION.WAIT].add(jnp.where(~has_target, 10.0, 0.0))

        pri = pri.at[ACTION.BOMB].add(jnp.where(at(dead_ends, x, y), 20.0, 0.0))
        at_target_crate = ~is_move & has_target & _touching_crate(field, x, y)
        pri = pri.at[ACTION.BOMB].add(jnp.where(at_target_crate, 20.0, 0.0))

        escape, corner = _bomb_escape(bombs, x, y, power)
        pri = pri + 30.0 * escape + 25.0 * corner

        on_bomb = at(bomb_grid, x, y)
        pri = pri.at[MOVES].add(
            jnp.where(on_bomb, 15.0 + jax.random.uniform(k_panic, MOVES.shape), 0.0)
        )

        best = jnp.argmax(jnp.where(valid, pri, -jnp.inf))

        return jnp.where(valid.any(), best, jnp.int32(ACTION.WAIT)).astype(jnp.int32), None, None
