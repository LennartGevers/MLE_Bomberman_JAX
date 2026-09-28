"""Actor-critic CNN after Gao et al. (arXiv:1905.01360), ``skynet955``."""

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from bomberman.game import ACTION

from .base import Network

N_ACTIONS = len(ACTION)


class SkynetCNNActorCriticNetwork(Network):
    """Actor-critic CNN after Gao et al. (arXiv:1905.01360), ``skynet955``.

    Four shared 3x3 convolutions. Each head then has two 1x1 convolutions and a fully connected layer.
    """

    has_value_head: ClassVar[bool] = True

    conv: list
    policy_head: list
    value_head: list

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        key: PRNGKeyArray,
        *,
        conv_channels: int = 64,  # Gao et al.
    ):
        channels, height, width = obs_shape
        inner_cells = (height - 2) * (width - 2)
        k1, k2, k3, k4, k5, k6, k7, k8 = jax.random.split(key, 8)
        self.conv = [
            eqx.nn.Conv2d(channels, conv_channels, kernel_size=3, padding=1, key=k1),
            jax.nn.relu,
            eqx.nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1, key=k2),
            jax.nn.relu,
            eqx.nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1, key=k3),
            jax.nn.relu,
            eqx.nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1, key=k4),
            jax.nn.relu,
        ]

        flat = 2 * inner_cells
        self.policy_head = [
            eqx.nn.Conv2d(conv_channels, 2, kernel_size=1, key=k5), jax.nn.relu,
            jnp.ravel,
            eqx.nn.Linear(flat, N_ACTIONS, key=k6),
        ]
        self.value_head = [
            eqx.nn.Conv2d(conv_channels, 2, kernel_size=1, key=k7), jax.nn.relu,
            jnp.ravel,
            eqx.nn.Linear(flat, 1, key=k8),
        ]

    def __call__(self, obs, key, *, inference: bool = False):
        x = obs[:, 1:-1, 1:-1]
        for layer in self.conv:
            x = layer(x)

        logits = x
        for layer in self.policy_head:
            logits = layer(logits)

        value = x
        for layer in self.value_head:
            value = layer(value)

        return logits, value[0]
