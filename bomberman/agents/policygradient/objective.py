"""Base class for policy-gradient objectives.

Subclasses compute the advantage. Returns, masking and the entropy bonus are handled here.
"""

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from bomberman.agents.base import Batch, _configure
from bomberman.agents.networks import Network

# Logit given to illegal actions. We avoid `-inf` because it produces `nan`
# gradients in the log-softmax.
_MASK_FILL = -1e9


def _discount(reward: Array, gamma: float) -> Array:
    """Discounted return at every tick of one game."""

    def scan(carry, r):
        g = r + gamma * carry
        return g, g

    _, out = jax.lax.scan(scan, jnp.float32(0.0), reward, reverse=True)
    return out


def _forward_key(key: PRNGKeyArray) -> PRNGKeyArray:
    """The part of ``key`` used by the network, e.g. for dropout."""
    # The agent splits its key the same way; ideally both would share one function.
    return jax.random.split(key)[0]


def _masked_log_policy(logits: Array, legal_mask: Array) -> Array:
    """Masked log-probabilities over actions, shape ``(N_ACTIONS,)``."""
    return jax.nn.log_softmax(jnp.where(legal_mask, logits, _MASK_FILL))


class Objective(eqx.Module):
    """The loss for policy-gradient training, e.g. REINFORCE or A2C."""

    needs_value: ClassVar[bool] = False

    gamma: Array = eqx.field(default_factory=lambda: jnp.asarray(0.99))
    entropy_coef: Array = eqx.field(default_factory=lambda: jnp.asarray(0.02))
    entropy_final: Array | None = eqx.field(default=None)

    def _advantage(
        self, returns: Array, value: Array | None, valid: Array, total: Array
    ) -> tuple[Array, Array, dict[str, Array]]:
        """Return ``(advantage, extra_loss, extra_metrics)`` for ``batch``."""
        raise NotImplementedError

    def _returns(self, reward: Array, value: Array | None, valid: Array) -> Array:
        """The return at every tick. By default the full Monte-Carlo return."""
        return jax.vmap(lambda r: _discount(r, self.gamma))(reward)

    def _entropy_coef(self, progress: Array) -> Array:
        coef = jnp.asarray(self.entropy_coef)
        if self.entropy_final is not None:
            coef = coef + progress * (self.entropy_final - coef)
        return coef

    def configure(self, **changes) -> Objective:
        """Return a copy with some fields replaced."""
        return _configure(self, changes)

    def __call__(self, network: Network, batch: Batch) -> tuple[Array, dict[str, Array]]:
        valid = batch.mask.astype(jnp.float32)
        total = jnp.maximum(valid.sum(), 1.0)

        fwd_keys = jax.vmap(jax.vmap(_forward_key))(batch.key)
        logits, value = jax.vmap(jax.vmap(network))(batch.obs.astype(jnp.float32), fwd_keys)
        returns = self._returns(batch.reward, value, valid)
        log_probs = _masked_log_policy(logits, batch.legal_mask)
        chosen = jnp.take_along_axis(log_probs, batch.action[..., None], axis=-1)[..., 0]

        advantage, extra_loss, extra_metrics = self._advantage(returns, value, valid, total)
        pg_loss = -(chosen * jax.lax.stop_gradient(advantage) * valid).sum() / total

        probs = jnp.exp(log_probs)
        entropy = -(probs * jnp.where(probs > 0, log_probs, 0.0)).sum(-1)
        mean_entropy = (entropy * valid).sum() / total

        coef = self._entropy_coef(batch.progress)
        loss = pg_loss + extra_loss - coef * mean_entropy
        return loss, {
            "loss": loss,
            "pg_loss": pg_loss,
            "entropy": mean_entropy,
            "entropy_coef": jnp.asarray(coef, jnp.float32),
            "return": (batch.reward * valid).sum(-1).mean(),
            "turns": valid.sum(-1).mean(),
            **extra_metrics,
        }
