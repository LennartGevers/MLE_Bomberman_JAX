"""Actor-critic CNN after Kartal et al. (arXiv:1907.10827)."""

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from bomberman.game import ACTION

from .base import Network

N_ACTIONS = len(ACTION)


class KartalCNNActorCriticNetwork(Network):
    """Actor-critic CNN after Kartal et al. (arXiv:1907.10827).

    Four 3x3 convolutions and a shared fully connected layer, followed by linear policy and value heads.
    """

    has_value_head: ClassVar[bool] = True

    conv: list
    mlp: list
    policy_head: eqx.nn.Linear
    value_head: eqx.nn.Linear

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        key: PRNGKeyArray,
        *,
        conv_channels: int = 32,  # Kartal et al.
        hidden: int = 128,        # Kartal et al.
    ):
        channels, height, width = obs_shape
        inner_cells = (height - 2) * (width - 2)
        k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)
        self.conv = [
            eqx.nn.Conv2d(channels, conv_channels, kernel_size=3, padding=1, key=k1),
            jax.nn.relu,
            eqx.nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1, key=k2),
            jax.nn.relu,
            eqx.nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1, key=k3),
            jax.nn.relu,
            eqx.nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1, key=k4),
            jax.nn.relu,
            jnp.ravel,
        ]
        
        flat = conv_channels * inner_cells
        self.mlp = [
            eqx.nn.Linear(flat, hidden, key=k5), jax.nn.relu,
        ]
        self.policy_head = eqx.nn.Linear(hidden, N_ACTIONS, key=k6)
        self.value_head  = eqx.nn.Linear(hidden, 1, key=k7)
        
    def __call__(self, obs, key, *, inference: bool = False):
        x = obs[:, 1:-1, 1:-1]
        for layer in self.conv + self.mlp:
            x = layer(x)
        return self.policy_head(x), self.value_head(x)[0]
