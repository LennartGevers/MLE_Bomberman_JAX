"""Quantities computed from one tick, used to build reward functions."""

import jax
import jax.numpy as jnp
from jaxtyping import Array
from pipefunc import pipefunc

from bomberman.game import FIELD, GameState
from bomberman.game.dynamics import _compute_blast, is_terminal
from bomberman.game.environment import EnvironmentState
from bomberman.game.settings import Settings

# Value meaning "no explosion will reach this tile". It is far larger than
# any real timer.
NO_DANGER = jnp.int32(1 << 30)


@pipefunc("coins_collected")
def coins_collected(prev_state: GameState, next_state: GameState) -> Array:
    """Coins each agent collected this tick."""
    newly = next_state.board.coins.collected & ~prev_state.board.coins.collected  # (C,)
    on_coin = jnp.all(
        prev_state.board.coins.pos[:, None, :]
        == next_state.board.agent_positions[None, :, :],
        axis=-1,
    )  # (C, N)
    return (newly[:, None] & on_coin).sum(axis=0).astype(jnp.float32)


@pipefunc("crates_destroyed")
def crates_destroyed(prev_state: GameState, next_state: GameState) -> Array:
    """The number of crates destroyed this tick."""
    before = (prev_state.board.field_map == FIELD.CRATE).sum()
    after = (next_state.board.field_map == FIELD.CRATE).sum()
    return (before - after).astype(jnp.float32)


@pipefunc("died")
def died(prev_state: GameState, next_state: GameState) -> Array:
    """1.0 for every agent that died this tick."""
    return (prev_state.alive & ~next_state.alive).astype(jnp.float32)


@pipefunc("loose_coin")
def loose_coin(state: GameState) -> Array:
    """Whether a visible coin is still on the board."""
    return state.board.coins.collectable(state.board.field_map).any()


@pipefunc("alive")
def alive(state: GameState) -> Array:
    """1.0 for every agent still alive."""
    return state.alive.astype(jnp.float32)


def blast_arrival(board: EnvironmentState, settings: Settings) -> Array:
    """Ticks until an explosion reaches each tile, or ``NO_DANGER``.

    Bombs that set off other bombs are taken into account.
    """
    bombs = board.bombs
    wall = board.field_map == FIELD.WALL
    n = bombs.pos.shape[0]

    blast = _compute_blast(bombs.pos, wall, settings.rules.bomb_power)  # (N, W, 2)
    fuse = jnp.where(bombs.active, bombs.timer, NO_DANGER)

    # reaches[i, j]: the explosion of bomb i would set off bomb j.
    reaches = (
        jnp.all(blast[:, None, :, :] == bombs.pos[None, :, None, :], axis=-1).any(axis=-1)
        & bombs.active[None, :]
    )

    def relax(fuse, _):
        triggered_by = jnp.where(reaches, fuse[:, None], NO_DANGER).min(axis=0)
        return jnp.minimum(fuse, triggered_by), None

    fuse, _ = jax.lax.scan(relax, fuse, None, length=n)

    grid = jnp.full(board.field_map.shape, NO_DANGER, dtype=jnp.int32)
    coords = blast.reshape(-1, 2)
    values = jnp.repeat(jnp.where(bombs.active, fuse, NO_DANGER), blast.shape[1])
    return grid.at[coords[:, 0], coords[:, 1]].min(values)


@pipefunc("danger_potential")
def danger_potential(state: GameState, settings: Settings) -> Array:
    """``-1`` for agents in the reach of a bomb, ``0`` otherwise."""
    board = state.board
    danger = blast_arrival(board, settings)
    apos = board.agent_positions
    hit = danger[apos[:, 0], apos[:, 1]] < NO_DANGER
    return jnp.where(hit, -1.0, 0.0).astype(jnp.float32)


@pipefunc("coin_potential")
def coin_potential(state: GameState) -> Array:
    """Grows towards ``0`` as an agent gets closer to the nearest coin."""
    board = state.board
    revealed = board.coins.collectable(board.field_map)  # (C,)
    apos = board.agent_positions  # (N, 2)
    coin_pos = board.coins.pos  # (C, 2)

    dist = jnp.abs(apos[:, None, :] - coin_pos[None, :, :]).sum(-1)  # (N, C)
    dist = jnp.where(revealed[None, :], dist, jnp.inf)
    nearest = jnp.min(dist, axis=1, initial=jnp.inf)  # (N,)

    height, width = board.field_map.shape
    span = jnp.float32((height - 1) + (width - 1))
    normalised = jnp.clip(nearest / span, 0.0, 1.0)
    return jnp.where(jnp.isfinite(nearest), -normalised, 0.0).astype(jnp.float32)


@pipefunc("shaping_potential")
def shaping_potential(
    state: GameState,
    settings: Settings,
    danger_weight: float = 1.0,
    coin_weight: float = 1.0,
) -> Array:
    """The shaping potential. It is zero when the game is over and for dead agents."""
    total = (
        danger_weight * danger_potential(state, settings)
        + coin_weight * coin_potential(state)
    )
    unsafe = state.alive & ~is_terminal(state, settings)
    return jnp.where(unsafe, total, 0.0)


@pipefunc("pbrs_delta")
def pbrs_delta(
    prev_state: GameState,
    next_state: GameState,
    settings: Settings,
    gamma: float = 0.99,
    danger_weight: float = 1.0,
    coin_weight: float = 1.0,
) -> Array:
    """The shaping reward ``gamma * phi(s') - phi(s)`` of every agent."""
    return (
        gamma * shaping_potential(next_state, settings, danger_weight, coin_weight)
        - shaping_potential(prev_state, settings, danger_weight, coin_weight)
    )


signal_funcs = [
    coins_collected,
    crates_destroyed,
    died,
    loose_coin,
    alive,
    danger_potential,
    coin_potential,
    shaping_potential,
    pbrs_delta,
]
