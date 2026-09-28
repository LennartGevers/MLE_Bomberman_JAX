"""The ``rule_based_agent`` of the original game, rewritten in JAX.

It walks to coins, crates or opponents along shortest paths, drops bombs when useful and runs away from explosions.
"""

from __future__ import annotations

from typing import NamedTuple

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
    NO_BOMB,
    self_position,
)
from .search import STAY, _shift, bfs_first_step

# The same constants as in the original agent: it remembers its last 20 tiles and
# 5 bomb spots, and counts as stuck if it was on the current tile more than twice.
_HISTORY_LEN = 20
_BOMB_HISTORY_LEN = 5
_LOOP_TRIGGER = 2
_LOOP_TIMER = 5
_NO_TILE = jnp.int32(-1)  # never matches a real position
class _State(NamedTuple):
    """What the agent remembers: its recent positions and bomb spots."""

    coord_history: Array  # (_HISTORY_LEN, 2) int32, oldest first
    bomb_history: Array  # (_BOMB_HISTORY_LEN, 2) int32, oldest first
    ignore_others_timer: Array  # () int32


def _push(history: Array, pos: Array) -> Array:
    """Append ``pos`` to a fixed-size history and drop the oldest entry."""
    return jnp.concatenate([history[1:], pos[None]], axis=0)


def _seen_count(history: Array, pos: Array) -> Array:
    return jnp.sum(jnp.all(history == pos, axis=-1))


def _bomb_cross_timer(bombs: Array, power: int) -> Array:
    """For each tile, the shortest timer of any bomb whose explosion would reach it."""
    fused = jnp.where(bombs > 0, bombs, NO_BOMB)
    cross = fused
    for dx, dy in DIRECTIONS.values():
        for r in range(1, power + 1):
            cross = jnp.minimum(cross, _shift(fused, dx * r, dy * r, NO_BOMB))
    return cross


def _free_neighbour_count(free: Array) -> Array:
    count = jnp.zeros_like(free, jnp.int32)
    for dx, dy in DIRECTIONS.values():
        count = count + _shift(free, dx, dy, False).astype(jnp.int32)
    return count


def _at(grid: Array, x: Array, y: Array) -> Array:
    h, w = grid.shape
    return grid[jnp.clip(x, 0, h - 1), jnp.clip(y, 0, w - 1)]


def _adjacent_any(mask: Array, x: Array, y: Array) -> Array:
    """Whether ``mask`` is set on any of the four neighbours of ``(x, y)``."""
    hit = jnp.bool_(False)
    for dx, dy in DIRECTIONS.values():
        hit = hit | _at(mask, x + dx, y + dy)
    return hit


def _bomb_escape(bombs: Array, x: Array, y: Array, power: int) -> tuple[Array, Array]:
    """Action weights for running away from a bomb in the same row or column."""
    bomb = bombs > 0
    escape = jnp.zeros(6, jnp.float32)
    corner = jnp.zeros(6, jnp.float32)

    # A bomb on our own tile has no "away" direction, so every direction is equally good.
    here = _at(bomb, x, y).astype(jnp.float32)
    for move in DIRECTIONS:
        corner = corner.at[move].add(here)

    for r in range(1, power + 1):
        below = _at(bomb, x, y + r).astype(jnp.float32)  # same column, larger y
        above = _at(bomb, x, y - r).astype(jnp.float32)
        right = _at(bomb, x + r, y).astype(jnp.float32)  # same row, larger x
        left = _at(bomb, x - r, y).astype(jnp.float32)

        escape = escape.at[ACTION.UP].add(below)
        escape = escape.at[ACTION.DOWN].add(above)
        corner = corner.at[ACTION.LEFT].add(below + above)
        corner = corner.at[ACTION.RIGHT].add(below + above)

        escape = escape.at[ACTION.LEFT].add(right)
        escape = escape.at[ACTION.RIGHT].add(left)
        corner = corner.at[ACTION.UP].add(right + left)
        corner = corner.at[ACTION.DOWN].add(right + left)

    return escape, corner


class RuleBasedAgent(Agent[_State]):
    """The rule-based agent of the original game. It does not learn."""

    power: int = eqx.field(static=True, default=3)
    features: FeatureShaping = ChannelStack()

    def init(self) -> _State:
        return _State(
            coord_history=jnp.full((_HISTORY_LEN, 2), _NO_TILE),
            bomb_history=jnp.full((_BOMB_HISTORY_LEN, 2), _NO_TILE),
            ignore_others_timer=jnp.int32(0),
        )

    def act(
        self,
        obs: Observation,
        legal_mask: Array,
        key: PRNGKeyArray,
        state: _State | None = None,
    ) -> tuple[Array, None, _State]:
        power = self.power
        planes = self.features(obs)
        field = planes[FIELD_CH]
        agents = planes[AGENTS]
        fire = planes[FIRE]
        bombs = planes[BOMBS]
        state = self.init() if state is None else state

        pos = self_position(agents)
        x, y = pos[0], pos[1]

        free = field == FIELD.EMPTY
        others = agents > 0
        bomb_grid = bombs > 0
        cross_timer = _bomb_cross_timer(bombs, power)

        # Detect loops: the agent is stuck if it was on this tile too often recently
        # (counted before adding the current tile, as in the original). When stuck,
        # `timer` is set and the agent stops chasing opponents for a few ticks,
        # since walking back and forth usually comes from a stand-off with one.
        stuck = _seen_count(state.coord_history, pos) > _LOOP_TRIGGER
        timer = jnp.where(
            stuck, jnp.int32(_LOOP_TIMER), jnp.maximum(state.ignore_others_timer - 1, 0)
        )
        coord_history = _push(state.coord_history, pos)

        # Avoid danger. The legal mask already rules out walls, bombs and occupied
        # tiles, but not tiles that are about to explode.
        offs = jnp.asarray([*DIRECTIONS.values(), (0, 0)], jnp.int32)  # moves, WAIT
        cx, cy = x + offs[:, 0], y + offs[:, 1]
        safe = (_at(fire, cx, cy) < 1) & (_at(cross_timer, cx, cy) > 0)
        valid = legal_mask & (
            jnp.ones(6, jnp.bool_)
            .at[MOVES].set(safe[:-1])
            .at[ACTION.WAIT].set(safe[-1])
        )
        # Do not bomb a tile again that was bombed recently.
        recently_bombed = _seen_count(state.bomb_history, pos) > 0
        valid = valid.at[ACTION.BOMB].set(valid[ACTION.BOMB] & ~recently_bombed)

        # Targets: coins, crates and dead ends, and opponents when hunting. The agent
        # hunts when there is nothing else to collect or when it is not stuck (`timer` is 0).
        coins = planes[COINS] > 0
        crates = field == FIELD.CRATE
        dead_ends = free & (_free_neighbour_count(free) == 1)
        hunting = (timer <= 0) | ~(coins.any() | crates.any())
        targets = ((coins | crates | dead_ends) | (others & hunting)) & ~bomb_grid
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

        pri = pri.at[ACTION.BOMB].add(jnp.where(_at(dead_ends, x, y), 20.0, 0.0))
        at_target_crate = ~is_move & has_target & _adjacent_any(crates, x, y)
        pri = pri.at[ACTION.BOMB].add(jnp.where(at_target_crate, 20.0, 0.0))
        touching_opponent = _adjacent_any(others, x, y)
        pri = pri.at[ACTION.BOMB].add(jnp.where(touching_opponent, 20.0, 0.0))

        escape, corner = _bomb_escape(bombs, x, y, power)
        pri = pri + 30.0 * escape + 25.0 * corner

        on_bomb = _at(bomb_grid, x, y)
        pri = pri.at[MOVES].add(
            jnp.where(on_bomb, 15.0 + jax.random.uniform(k_panic, MOVES.shape), 0.0)
        )

        best = jnp.argmax(jnp.where(valid, pri, -jnp.inf))
        action = jnp.where(valid.any(), best, jnp.int32(ACTION.WAIT)).astype(jnp.int32)

        bomb_history = jnp.where(
            action == ACTION.BOMB, _push(state.bomb_history, pos), state.bomb_history
        )
        return action, None, _State(coord_history, bomb_history, timer)
