"""The board and the fixed-size tables of coins, bombs and explosions."""

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from bomberman.game.settings import Settings


class FIELD(IntEnum):
    WALL = -1
    EMPTY = 0
    CRATE = 1


# The tiles around a starting corner that are kept free of crates.
_PLUS = jnp.asarray([(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)], dtype=jnp.int32)

# Where hidden coins are placed in an observation (see `Coins.redacted`). The
# top-left tile is always a wall, so no real coin can be there, and a coin on
# this tile is never collectable and never shows up on any map.
HIDDEN_COIN = jnp.zeros(2, dtype=jnp.int32)


def blast_width(power: int) -> int:
    """The number of tiles a single explosion can cover."""
    return 1 + 4 * power


class Coins(eqx.Module):
    pos: Array        # (C, 2) int32, fixed for the whole game
    collected: Array  # (C,) bool

    def revealed(self, field_map: Array) -> Array:
        """Coins that are not under a crate, collected or not."""
        return field_map[self.pos[:, 0], self.pos[:, 1]] == FIELD.EMPTY

    def collectable(self, field_map: Array) -> Array:
        """Coins that can be picked up: not collected yet and not under a crate."""
        return self.revealed(field_map) & ~self.collected

    def hidden(self, field_map: Array) -> Array:
        """Coins still buried under a crate."""
        return ~self.revealed(field_map) & ~self.collected

    def redacted(self, field_map: Array) -> Coins:
        """A copy in which every hidden coin is moved to ``HIDDEN_COIN``."""
        return Coins(
            pos=jnp.where(self.hidden(field_map)[:, None], HIDDEN_COIN, self.pos),
            collected=self.collected,
        )

    def map(self, field_map: Array) -> Array:
        return jnp.zeros_like(field_map).at[
            self.pos[:, 0], self.pos[:, 1]
        ].set(
            jnp.where(self.collectable(field_map), 1, 0)
        )


class Bombs(eqx.Module):
    pos: Array     # (N, 2) int32, position of agent i's bomb (only meaningful if active)
    timer: Array   # (N,) int32, ticks until the bomb explodes
    active: Array  # (N,) bool

    @property
    def detonating(self) -> Array:
        return self.active & (self.timer == 1)

    def map(self, field_map: Array) -> Array:
        """A map with each bomb's remaining timer on its tile and ``0`` everywhere else."""
        return jnp.zeros_like(field_map).at[
            self.pos[:, 0], self.pos[:, 1]
        ].max(
            jnp.where(self.active, self.timer, 0)
        )


class Explosions(eqx.Module):
    blast: Array   # (N, blast_width, 2) int32, tiles covered by agent i's explosion
                   # (where a wall cuts a ray short, the bomb's own tile fills the gap)
    timer: Array   # (N,) int32, ticks the explosion stays deadly
    active: Array  # (N,) bool

    def map(self, field_map: Array) -> Array:
        explosions_per_blast = self.blast.shape[1]
        coords = self.blast.reshape(-1, 2)
        explosion = jnp.where(jnp.repeat(self.active, explosions_per_blast), jnp.repeat(self.timer, explosions_per_blast), 0)

        return jnp.zeros_like(field_map).at[coords[:, 0], coords[:, 1]].set(explosion)

    def covers(self, positions: Array) -> Array:
        """For each position, which active explosions cover it. Shape ``(len(positions), N)``."""
        match = jnp.all(
            positions[:, None, None, :] == self.blast[None, :, :, :], axis=-1
        ).any(axis=-1)
        return match & self.active[None, :]


class EnvironmentState(eqx.Module):
    field_map: Array          # (H, W) int32, walls, crates and empty tiles
    agent_positions: Array    # (N, 2) int32
    coins: Coins
    bombs: Bombs
    explosions: Explosions


def init_environment(settings: Settings, key: Array) -> EnvironmentState:
    scenario = settings.scenario
    crate_key, coin_key, agent_key = jax.random.split(key, 3)

    size = scenario.size
    n = scenario.agent_count
    width = blast_width(settings.rules.bomb_power)

    field_map = jnp.where(
        jax.random.uniform(crate_key, (size, size)) < scenario.crate_density,
        jnp.int32(FIELD.CRATE),
        jnp.int32(FIELD.EMPTY),
    )
    field_map = _attach_walls(field_map)

    starts = _start_positions(size)
    field_map = _clear_start_positions(field_map, starts)

    return EnvironmentState(
        field_map=field_map,
        agent_positions=_create_agent_positions(starts, n, agent_key),
        coins=_create_coins(field_map, scenario.coin_count, coin_key),
        bombs=Bombs(
            pos=jnp.zeros((n, 2), jnp.int32),
            timer=jnp.zeros(n, jnp.int32),
            active=jnp.zeros(n, jnp.bool_),
        ),
        explosions=Explosions(
            blast=jnp.zeros((n, width, 2), jnp.int32),
            timer=jnp.zeros(n, jnp.int32),
            active=jnp.zeros(n, jnp.bool_),
        ),
    )


def _start_positions(size: int) -> Array:
    return jnp.asarray(
        [(1, 1), (1, size - 2), (size - 2, 1), (size - 2, size - 2)],
        dtype=jnp.int32,
    )


def _attach_walls(field_map: Array) -> Array:
    cols, rows = field_map.shape
    field_map = (
        field_map.at[0, :].set(FIELD.WALL)
        .at[-1, :].set(FIELD.WALL)
        .at[:, 0].set(FIELD.WALL)
        .at[:, -1].set(FIELD.WALL)
    )
    xs, ys = jnp.meshgrid(jnp.arange(cols), jnp.arange(rows), indexing="ij")
    pillars = ((xs + 1) * (ys + 1)) % 2 == 1
    return jnp.where(pillars, jnp.int32(FIELD.WALL), field_map)


def _clear_start_positions(field_map: Array, starts: Array) -> Array:
    tiles = (starts[:, None, :] + _PLUS[None, :, :]).reshape(-1, 2)
    xs, ys = tiles[:, 0], tiles[:, 1]
    current = field_map[xs, ys]
    return field_map.at[xs, ys].set(
        jnp.where(current == FIELD.CRATE, jnp.int32(FIELD.EMPTY), current)
    )


def _create_coins(field_map: Array, coin_count: int, key: Array) -> Coins:
    """Place ``coin_count`` coins, under crates where possible."""
    eligible = (field_map == FIELD.CRATE) | (field_map == FIELD.EMPTY)
    noise = jax.random.uniform(key, field_map.shape)
    priority = jnp.where(field_map == FIELD.EMPTY, 1.0, 0.0) + noise
    priority = jnp.where(eligible, priority, jnp.inf)

    idx = jnp.argsort(priority.ravel())[:coin_count]
    pos = jnp.stack(jnp.unravel_index(idx, field_map.shape), axis=-1).astype(jnp.int32)
    return Coins(pos=pos, collected=jnp.zeros(coin_count, jnp.bool_))


def _create_agent_positions(starts: Array, agent_count: int, key: Array) -> Array:
    return jax.random.permutation(key, starts)[:agent_count]
