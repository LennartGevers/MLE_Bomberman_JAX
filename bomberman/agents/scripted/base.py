import jax.numpy as jnp
from jaxtyping import Array

from bomberman.agents.features import AGENTS, BOMBS, COINS, COOLDOWN, FIELD_CH, FIRE
from bomberman.game.const import DIRECTIONS

from .search import _shift

# The scripted agents read the planes of `ChannelStack` by index.
__all__ = [
    "AGENTS",
    "BOMBS",
    "COINS",
    "COOLDOWN",
    "FIELD_CH",
    "FIRE",
    "MOVES",
    "NO_BOMB",
    "at",
    "bomb_cross_timer",
    "free_neighbour_count",
    "self_position",
]

# Timer value meaning "no bomb reaches this tile", as in the original `bomb_map`.
NO_BOMB = 5

# Action ids of the four moves.
MOVES = jnp.asarray(list(DIRECTIONS), jnp.int32)


def self_position(agents: Array) -> Array:
    """Our own position, marked with ``-1`` in the agents plane."""
    flat = jnp.argmax((agents == -1).ravel())
    x, y = jnp.unravel_index(flat, agents.shape)
    return jnp.stack([x, y]).astype(jnp.int32)

def bomb_cross_timer(bombs: Array, power: int) -> Array:
    """For each tile, the shortest timer of any bomb whose explosion would reach it."""
    fused = jnp.where(bombs > 0, bombs, NO_BOMB)
    cross = fused
    for dx, dy in DIRECTIONS.values():
        for r in range(1, power + 1):
            cross = jnp.minimum(cross, _shift(fused, dx * r, dy * r, NO_BOMB))
    return cross


def free_neighbour_count(free: Array) -> Array:
    count = jnp.zeros_like(free, jnp.int32)
    for dx, dy in DIRECTIONS.values():
        count = count + _shift(free, dx, dy, False).astype(jnp.int32)
    return count


def at(grid: Array, x: Array, y: Array) -> Array:
    h, w = grid.shape
    return grid[jnp.clip(x, 0, h - 1), jnp.clip(y, 0, w - 1)]
