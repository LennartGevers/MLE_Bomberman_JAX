"""REINFORCE with normalised Monte-Carlo returns."""

from __future__ import annotations

import jax.numpy as jnp

from .objective import Objective


class ReinforceObjective(Objective):
    """REINFORCE: the Monte-Carlo return, normalised over the batch, is the advantage."""

    def _advantage(self, returns, value, valid, total):
        mean = (returns * valid).sum() / total
        std = jnp.sqrt((((returns - mean) ** 2) * valid).sum() / total + 1e-8)
        # Clip the normalised advantage. If all games look alike, the standard
        # deviation is close to 0 and the advantages would become huge.
        advantage = jnp.clip((returns - mean) / std, -10.0, 10.0)
        return advantage, jnp.float32(0.0), {}
