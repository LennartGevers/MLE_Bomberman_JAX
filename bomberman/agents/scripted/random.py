"""An agent that picks a random legal action."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from bomberman.game.const import ACTION

from ..base import N_ACTIONS, Agent


def action_mask(forbid_bomb: bool) -> Array:
    """All actions, without BOMB if ``forbid_bomb`` is set."""
    mask = jnp.ones(N_ACTIONS, jnp.bool_)
    if forbid_bomb:
        mask = mask.at[ACTION.BOMB].set(False)
    return mask

class RandomAgent(Agent[None]):
    """Picks a random legal action."""

    forbid_bomb: bool = eqx.field(default=False, static=True)

    def act(self, obs, legal_mask, key, state=None):

        logits = jnp.where(legal_mask & action_mask(self.forbid_bomb), 0.0, -jnp.inf)
        return jax.random.categorical(key, logits).astype(jnp.int32), None, state
