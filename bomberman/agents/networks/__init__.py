"""Networks that map an observation to action logits and, for actor-critics, a value."""

from bomberman.agents.networks.base import Network
from bomberman.agents.networks.cnn import (
    CNNActorCriticNetwork,
    CNNNetwork,
    DeepResNetwork,
    DropoutCNNNetwork,
)
from bomberman.agents.networks.kartal_cnn import KartalCNNActorCriticNetwork
from bomberman.agents.networks.skynet_cnn import SkynetCNNActorCriticNetwork

__all__ = [
    "CNNActorCriticNetwork",
    "CNNNetwork",
    "DeepResNetwork",
    "DropoutCNNNetwork",
    "KartalCNNActorCriticNetwork",
    "Network",
    "SkynetCNNActorCriticNetwork",
]
