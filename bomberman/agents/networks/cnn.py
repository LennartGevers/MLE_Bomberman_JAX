"""CNN policy and actor-critic networks."""

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from bomberman.game import ACTION

from .base import Network

N_ACTIONS = len(ACTION)


class CNNNetwork(Network):
    """Two convolutions and a two-layer MLP. The wall around the board is cropped first."""

    layers: list

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        key: PRNGKeyArray,
        *,
        conv_channels: int = 32,
        hidden: int = 256,
    ):
        channels, height, width = obs_shape
        inner_cells = (height - 2) * (width - 2)
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.layers = [
            eqx.nn.Conv2d(channels, conv_channels, kernel_size=3, padding=1, key=k1),
            jax.nn.relu,
            eqx.nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1, key=k2),
            jax.nn.relu,
            jnp.ravel,
            eqx.nn.Linear(conv_channels * inner_cells, hidden, key=k3),
            jax.nn.relu,
            eqx.nn.Linear(hidden, N_ACTIONS, key=k4),
        ]

    def __call__(self, obs, key, *, inference: bool = False):
        x = obs[:, 1:-1, 1:-1]
        for layer in self.layers:
            x = layer(x)
        return x, None


class _ResBlock(eqx.Module):
    """Residual block: GroupNorm, ReLU and convolution, twice."""

    norm1: eqx.nn.GroupNorm
    conv1: eqx.nn.Conv2d
    norm2: eqx.nn.GroupNorm
    conv2: eqx.nn.Conv2d

    def __call__(self, x: Array) -> Array:
        h = self.conv1(jax.nn.relu(self.norm1(x)))
        h = self.conv2(jax.nn.relu(self.norm2(h)))
        return x + h


class DeepResNetwork(Network):
    """A stack of residual blocks, global average pooling and a small MLP."""

    stem: eqx.nn.Conv2d
    blocks: list
    out_norm: eqx.nn.GroupNorm
    head: list

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        key: PRNGKeyArray,
        *,
        channels: int = 48,
        hidden: int = 128,
        n_blocks: int = 3,
        groups: int = 8,
    ):
        channels_in, _, _ = obs_shape
        keys = jax.random.split(key, 3 + 2 * n_blocks)
        self.stem = eqx.nn.Conv2d(channels_in, channels, kernel_size=3, padding=1, key=keys[0])
        self.blocks = [
            _ResBlock(
                eqx.nn.GroupNorm(groups, channels),
                eqx.nn.Conv2d(channels, channels, kernel_size=3, padding=1, key=keys[1 + 2 * i]),
                eqx.nn.GroupNorm(groups, channels),
                eqx.nn.Conv2d(channels, channels, kernel_size=3, padding=1, key=keys[2 + 2 * i]),
            )
            for i in range(n_blocks)
        ]
        self.out_norm = eqx.nn.GroupNorm(groups, channels)
        self.head = [
            eqx.nn.Linear(channels, hidden, key=keys[-2]),
            jax.nn.relu,
            eqx.nn.Linear(hidden, N_ACTIONS, key=keys[-1]),
        ]

    def __call__(self, obs, key, *, inference: bool = False):
        x = self.stem(obs)
        for block in self.blocks:
            x = block(x)
        x = jax.nn.relu(self.out_norm(x))
        x = x.mean(axis=(1, 2))  # global average pooling, shape (channels,)
        for layer in self.head:
            x = layer(x)
        return x, None


class DropoutCNNNetwork(Network):
    """A wider CNN with dropout in the MLP. Dropout is off when ``inference=True``."""

    layers: list
    n_dropout: int = eqx.field(static=True)

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        key: PRNGKeyArray,
        *,
        conv_channels: int = 48,
        hidden: int = 256,
        p: float = 0.1,
    ):
        channels, height, width = obs_shape
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.layers = [
            eqx.nn.Conv2d(channels, conv_channels, kernel_size=3, padding=1, key=k1),
            jax.nn.relu,
            eqx.nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1, key=k2),
            jax.nn.relu,
            jnp.ravel,
            eqx.nn.Linear(conv_channels * height * width, hidden, key=k3),
            jax.nn.relu,
            eqx.nn.Dropout(p),
            eqx.nn.Linear(hidden, N_ACTIONS, key=k4),
        ]
        self.n_dropout = sum(isinstance(layer, eqx.nn.Dropout) for layer in self.layers)

    def __call__(self, obs, key, *, inference: bool = False):
        drop_keys = (
            [None] * self.n_dropout
            if inference
            else list(jax.random.split(key, self.n_dropout))
        )
        di, x = 0, obs
        for layer in self.layers:
            if isinstance(layer, eqx.nn.Dropout):
                x = layer(x, key=drop_keys[di], inference=inference)
                di += 1
            else:
                x = layer(x)
        return x, None


class CNNActorCriticNetwork(Network):
    """The layers of ``CNNNetwork`` with separate policy and value heads."""

    has_value_head: ClassVar[bool] = True

    trunk: list
    policy_head: list
    value_head: list

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        key: PRNGKeyArray,
        *,
        conv_channels: int = 32,
        hidden: int = 256,
    ):
        channels, height, width = obs_shape
        inner_cells = (height - 2) * (width - 2)
        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
        self.trunk = [
            eqx.nn.Conv2d(channels, conv_channels, kernel_size=3, padding=1, key=k1),
            jax.nn.relu,
            eqx.nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1, key=k2),
            jax.nn.relu,
            jnp.ravel,
        ]
        flat = conv_channels * inner_cells
        self.policy_head = [
            eqx.nn.Linear(flat, hidden, key=k3),
            jax.nn.relu,
            eqx.nn.Linear(hidden, N_ACTIONS, key=k4),
        ]
        self.value_head = [
            eqx.nn.Linear(flat, hidden, key=k5),
            jax.nn.relu,
            eqx.nn.Linear(hidden, 1, key=k6),
        ]

    def __call__(self, obs, key, *, inference: bool = False):
        x = obs[:, 1:-1, 1:-1]
        for layer in self.trunk:
            x = layer(x)

        logits = x
        for layer in self.policy_head:
            logits = layer(logits)

        value = x
        for layer in self.value_head:
            value = layer(value)

        return logits, value[0]
