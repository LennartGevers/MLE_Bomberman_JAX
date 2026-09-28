"""Breadth-first search on the board, for the scripted agents."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array

from bomberman.game.const import ACTION, DIRECTIONS

STAY = int(ACTION.WAIT)
_BIG = jnp.int32(1 << 20)


def _shift(grid: Array, dx: int, dy: int, fill) -> Array:
    """Shift ``grid`` by ``(dx, dy)`` and fill the uncovered edge with ``fill``."""
    out = jnp.roll(grid, (dx, dy), axis=(0, 1))
    if dx > 0:
        out = out.at[:dx, :].set(fill)
    elif dx < 0:
        out = out.at[dx:, :].set(fill)
    if dy > 0:
        out = out.at[:, :dy].set(fill)
    elif dy < 0:
        out = out.at[:, dy:].set(fill)
    return out


def _default_iters(h: int, w: int) -> int:
    # Enough steps for any shortest path on a Bomberman board.
    return 2 * (h + w)


def bfs_field(
    free_space: Array, start: Array, *, max_iter: int | None = None
) -> tuple[Array, Array]:
    """Breadth-first search from ``start`` over the free tiles.

    Returns the distance to every tile and the first move on the way there, both of shape ``(H, W)``.
    """
    h, w = free_space.shape
    iters = _default_iters(h, w) if max_iter is None else max_iter
    sx, sy = start[0], start[1]

    is_start = jnp.zeros((h, w), jnp.bool_).at[sx, sy].set(True)
    dist = jnp.where(is_start, jnp.int32(0), _BIG)
    first = jnp.full((h, w), jnp.int32(STAY))

    def relax(_, carry):
        dist, first = carry
        for move, (dx, dy) in DIRECTIONS.items():
            pred_dist = _shift(dist, dx, dy, _BIG)
            pred_first = _shift(first, dx, dy, jnp.int32(STAY))
            pred_is_start = _shift(is_start, dx, dy, False)
            cand = pred_dist + 1
            better = free_space & (cand < dist)
            dist = jnp.where(better, cand, dist)
            fm = jnp.where(pred_is_start, jnp.int32(move), pred_first)
            first = jnp.where(better, fm, first)
        return dist, first

    return jax.lax.fori_loop(0, iters, relax, (dist, first))


def manhattan_field(targets: Array, *, max_iter: int | None = None) -> Array:
    """Manhattan distance from every tile to the nearest target."""
    h, w = targets.shape
    iters = _default_iters(h, w) if max_iter is None else max_iter
    md = jnp.where(targets, jnp.int32(0), _BIG)

    def relax(_, md):
        for dx, dy in DIRECTIONS.values():
            md = jnp.minimum(md, _shift(md, dx, dy, _BIG) + 1)
        return md

    return jax.lax.fori_loop(0, iters, relax, md)


def bfs_first_step(
    free_space: Array, start: Array, targets: Array, *, max_iter: int | None = None
) -> Array:
    """The first move from ``start`` towards the nearest reachable target.

    If no target can be reached, the agent heads for the reachable tile closest to one.
    """
    dist, first = bfs_field(free_space, start, max_iter=max_iter)
    md = manhattan_field(targets, max_iter=max_iter)

    reachable = jnp.where(targets, dist, _BIG)
    any_reachable = reachable.min() < _BIG
    best_reachable = jnp.argmin(reachable.ravel())

    # If the target itself cannot be entered (a crate or another agent), staying
    # put and stepping towards it can tie. The original agent then prefers the
    # tile it found later, i.e. the one farther away. A tiny bonus for larger
    # `dist` does the same: it only decides between equal totals, so the agent
    # moves instead of standing still.
    explored = dist < _BIG
    fallback = (dist + md).astype(jnp.float32) - 1e-3 * dist.astype(jnp.float32)
    fallback = jnp.where(explored, fallback, jnp.float32(_BIG))
    best_fallback = jnp.argmin(fallback.ravel())

    best = jnp.where(any_reachable, best_reachable, best_fallback)
    move = first.ravel()[best]
    return jnp.where(targets.any(), move, jnp.int32(STAY)).astype(jnp.int32)
