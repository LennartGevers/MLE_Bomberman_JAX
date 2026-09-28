"""Advantage actor-critic after Mnih et al. (2016), arXiv:1602.01783."""

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from .objective import Objective


def _n_step_returns(
    reward: Array, value: Array, valid: Array, gamma: float, n: int
) -> Array:
    """n-step returns, shape ``(B, R)``. No gradient flows through the bootstrapped value."""
    rounds = reward.shape[1]
    pad = ((0, 0), (0, n))
    reward = jnp.pad(reward, pad)
    bootstrap = jnp.pad(jax.lax.stop_gradient(value) * valid, pad)

    target = jnp.zeros_like(reward[:, :rounds])
    for k in range(n):
        target = target + gamma**k * reward[:, k : k + rounds]
    return target + gamma**n * bootstrap[:, n : n + rounds]


class A2CObjective(Objective):
    """Advantage actor-critic with n-step bootstrapped returns.

    With ``n_steps=None`` the full Monte-Carlo return is used. The network needs a value head.
    """

    needs_value: ClassVar[bool] = True

    value_coef: float = eqx.field(static=True, default=0.5)
    n_steps: int | None = eqx.field(static=True, default=5)

    def __check_init__(self):
        if self.n_steps is not None and self.n_steps < 1:
            raise ValueError(f"n_steps must be >= 1 or None, got {self.n_steps}")

    def _returns(self, reward, value, valid):
        if self.n_steps is None:
            return super()._returns(reward, value, valid)
        return _n_step_returns(reward, value, valid, self.gamma, self.n_steps)

    def _advantage(self, returns, value, valid, total):
        advantage = returns - value
        # No stop-gradient on `advantage` here: its gradient with respect to V is the
        # usual value-loss gradient. The bootstrapped value inside `returns` is
        # already stopped, so this is the standard semi-gradient TD update.
        value_loss = 0.5 * ((advantage**2) * valid).sum() / total
        return_mean = (returns * valid).sum() / total
        return_var = (((returns - return_mean) ** 2) * valid).sum() / total
        explained_variance = 1.0 - value_loss * 2.0 / jnp.maximum(return_var, 1e-8)
        extra_metrics = {"value_loss": value_loss, "explained_variance": explained_variance}
        return advantage, self.value_coef * value_loss, extra_metrics
