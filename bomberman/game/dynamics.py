"""The rules of the game. One call to ``step`` plays one tick.

First all actions are carried out in a random turn order, then coins, bombs and explosions are updated.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jaxtyping import Array

from bomberman.game.const import ACTION, DIRECTION_ARR, DIRECTIONS, SWITCH_BRANCH, TRUE
from bomberman.game.environment import FIELD, Bombs, EnvironmentState, Explosions
from bomberman.game.settings import Settings

if TYPE_CHECKING:
    from bomberman.game.game import GameState


def legal_action_mask(state: GameState, agent_id: Array) -> Array:
    """Boolean mask of the actions ``agent_id`` may take this tick."""
    legal_actions = jnp.concatenate([
        _check_movement(state, agent_id),
        jnp.expand_dims(_check_cooldown(state, agent_id), axis=0),
        jnp.expand_dims(TRUE, axis=0),  # WAIT is always legal
    ])

    return _check_health(legal_actions, state, agent_id)


def _check_movement(state: GameState, agent_id: Array) -> Array:
    x, y = state.board.agent_positions[agent_id]
    return jnp.asarray([
        _is_tile_free(state, jnp.asarray((x + dx, y + dy)))
        for dx, dy in DIRECTIONS.values()
    ])


def _is_tile_free(state: GameState, tile: Array) -> Array:
    """Whether an agent may step onto ``tile``: empty ground without a bomb or a living agent."""
    board = state.board
    x, y = tile[0], tile[1]

    empty = board.field_map[x, y] == FIELD.EMPTY
    no_bomb = ~(
        (board.bombs.pos[:, 0] == x)
        & (board.bombs.pos[:, 1] == y)
        & board.bombs.active
    ).any()
    unoccupied = ~(
        (board.agent_positions[:, 0] == x)
        & (board.agent_positions[:, 1] == y)
        & state.alive
    ).any()
    return empty & no_bomb & unoccupied


def _check_cooldown(state: GameState, agent_id: Array) -> Array:
    return state.bomb_cooldown[agent_id] == 0


def _check_health(actions: Array, state: GameState, agent_id: Array) -> Array:
    return jax.lax.cond(
        state.alive[agent_id],
        lambda: actions,
        lambda: jnp.zeros_like(actions).at[ACTION.WAIT].set(TRUE),
    )


def step(state: GameState, actions: Array, settings: Settings) -> GameState:
    """Play one tick, given one action per agent."""
    return _process_step(_perform_actions(state, actions, settings), settings)


def _perform_actions(
    state: GameState, actions: Array, settings: Settings
) -> GameState:
    """Carry out the actions in turn order. Dead agents do nothing."""
    for i in range(state.num_agents):
        agent = state.turn_order[i]
        action = jnp.where(state.alive[agent], actions[agent], jnp.int32(ACTION.WAIT))
        state = _perform_action(state, agent, action, settings)
    return state


def _perform_action(
    state: GameState, agent_id: Array, action: Array, settings: Settings
) -> GameState:
    return jax.lax.switch(
        SWITCH_BRANCH[action],
        [
            lambda: _move(state, action, agent_id),
            lambda: _plant(state, agent_id, settings),
            lambda: state,
        ],
    )


def _move(state: GameState, action: Array, agent_id: Array) -> GameState:
    current = state.board.agent_positions[agent_id]
    target = current + DIRECTION_ARR[action]
    # The tile was free when the agent chose its action, but an agent that moved
    # earlier this tick may have taken it. In that case the agent stays put.
    new_position = jnp.where(_is_tile_free(state, target), target, current)
    return dataclasses.replace(
        state,
        board=dataclasses.replace(
            state.board,
            agent_positions=state.board.agent_positions.at[agent_id].set(new_position),
        ),
    )


def _plant(state: GameState, agent_id: Array, settings: Settings) -> GameState:
    bombs = state.board.bombs
    position = state.board.agent_positions[agent_id]
    can_plant = (state.bomb_cooldown[agent_id] == 0) & state.alive[agent_id]
    fuse = settings.rules.bomb_timer + 1

    new_bombs = Bombs(
        pos=bombs.pos.at[agent_id].set(
            jnp.where(can_plant, position, bombs.pos[agent_id])
        ),
        timer=bombs.timer.at[agent_id].set(
            jnp.where(can_plant, fuse, bombs.timer[agent_id])
        ),
        active=bombs.active.at[agent_id].set(can_plant | bombs.active[agent_id]),
    )
    bomb_cooldown = state.bomb_cooldown.at[agent_id].set(
        jnp.where(can_plant, settings.rules.bomb_cooldown, state.bomb_cooldown[agent_id])
    )
    return dataclasses.replace(
        state,
        board=dataclasses.replace(state.board, bombs=new_bombs),
        bomb_cooldown=bomb_cooldown,
    )


def _process_step(state: GameState, settings: Settings) -> GameState:
    """Everything that happens after the actions: coins, bombs, explosions, deaths and scores."""
    board = state.board
    num_agents = state.num_agents
    alive_before = state.alive
    apos = board.agent_positions

    # 1. Collect coins. As in the original game, this happens before the bombs go off.
    collectable = board.coins.collectable(board.field_map)                  # (C,)
    on_coin = jnp.all(board.coins.pos[:, None, :] == apos[None, :, :], axis=-1)  # (C, N)
    picked_by = on_coin & collectable[:, None] & alive_before[None, :]      # (C, N)
    coins = dataclasses.replace(
        board.coins, collected=board.coins.collected | picked_by.any(axis=1)
    )
    coin_reward = picked_by.sum(axis=0).astype(jnp.float32) * settings.rules.reward_coin

    # 2. Age existing explosions.
    aged_timer = jnp.maximum(board.explosions.timer - 1, 0)
    aged_active = board.explosions.active & (aged_timer > 0)

    # 3. Detonate bombs whose fuse runs out this tick.
    wall = board.field_map == FIELD.WALL
    detonating = board.bombs.detonating
    fresh_blast = _compute_blast(board.bombs.pos, wall, settings.rules.bomb_power)

    explosions = Explosions(
        blast=jnp.where(detonating[:, None, None], fresh_blast, board.explosions.blast),
        timer=jnp.where(detonating, settings.rules.explosion_timer, aged_timer),
        active=detonating | aged_active,
    )
    bombs = Bombs(
        pos=board.bombs.pos,
        timer=jnp.where(
            board.bombs.active & ~detonating, board.bombs.timer - 1, board.bombs.timer
        ),
        active=board.bombs.active & ~detonating,
    )

    # Crates inside an explosion are destroyed.
    danger = explosions.map(board.field_map)
    field_map = jnp.where(
        (danger > 0) & (board.field_map == FIELD.CRATE),
        jnp.int32(FIELD.EMPTY),
        board.field_map,
    )

    # 4. Kill agents standing in an explosion. `hit[a, e]` is True if agent a stands in explosion e.
    hit = explosions.covers(apos)
    in_fire = hit.any(axis=1)
    died = alive_before & in_fire
    alive = alive_before & ~in_fire

    # 5. Scores: points for collected coins and killed opponents (killing yourself earns nothing).
    #    As in the original game, every bomb whose explosion hit a dying agent gets the kill.
    #    The points are added to the running total, so `state.score` is the scoreboard.
    agent_ids = jnp.arange(num_agents)
    credit = hit & died[:, None] & (agent_ids[None, :] != agent_ids[:, None])
    kill_reward = credit.sum(axis=0).astype(jnp.float32) * settings.rules.reward_kill
    score = state.score + coin_reward + kill_reward

    new_board = EnvironmentState(
        field_map=field_map,
        agent_positions=apos,
        coins=coins,
        bombs=bombs,
        explosions=explosions,
    )

    return dataclasses.replace(
        state,
        board=new_board,
        alive=alive,
        score=score,
        bomb_cooldown=jnp.maximum(state.bomb_cooldown - 1, 0),
        turn_order=jax.random.permutation(state.key, jnp.arange(num_agents)),
        step_count=state.step_count + 1,
    )


def _compute_blast(bomb_pos: Array, wall: Array, power: int) -> Array:
    """The tiles each bomb's explosion covers, shape ``(N, 1 + 4 * power, 2)``.

    The explosion covers the bomb's tile and extends ``power`` tiles in each direction until it hits a wall.
    """
    h, w = wall.shape
    coords = [bomb_pos]
    for dx, dy in DIRECTIONS.values():
        blocked = jnp.zeros(bomb_pos.shape[0], jnp.bool_)
        for i in range(1, power + 1):
            tile = bomb_pos + jnp.asarray((dx * i, dy * i), jnp.int32)
            tx = jnp.clip(tile[:, 0], 0, h - 1)
            ty = jnp.clip(tile[:, 1], 0, w - 1)
            blocked = blocked | wall[tx, ty]
            coords.append(jnp.where(blocked[:, None], bomb_pos, tile))
    return jnp.stack(coords, axis=1)


def is_terminal(state: GameState, settings: Settings) -> Array:
    board = state.board
    n_alive = state.alive.sum()

    nothing_left = (
        ((board.field_map == FIELD.CRATE).sum() == 0)
        & ~board.coins.collectable(board.field_map).any()
        & ~board.bombs.active.any()
        & ~board.explosions.active.any()
    )
    stalemate = (n_alive <= 1) & nothing_left
    timeout = state.step_count >= settings.rules.max_steps
    return (n_alive == 0) | stalemate | timeout
