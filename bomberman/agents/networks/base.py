"""The ``Network`` base class."""

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
from jaxtyping import Array, PRNGKeyArray

from bomberman.agents.features import ChannelStack, FeatureShaping


class Network(eqx.Module):
    """Maps an observation to action logits and, if it has a critic, a value."""

    has_value_head: ClassVar[bool] = False

    def __call__(
        self, obs: Array, key: PRNGKeyArray, *, inference: bool = False
    ) -> tuple[Array, Array | None]:
        """Return ``(logits, value)``. Illegal actions are not masked, and ``value`` may be ``None``."""
        raise NotImplementedError

    @classmethod
    def for_settings(
        cls,
        settings,
        key: PRNGKeyArray,
        *,
        features: FeatureShaping | None = None,
        **kwargs,
    ) -> Network:
        """Build a network whose input fits ``features`` on a board with ``settings``."""
        features = features if features is not None else ChannelStack()
        return cls(features.shape(settings), key, **kwargs)
