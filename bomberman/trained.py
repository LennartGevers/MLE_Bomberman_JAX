"""Load the two final agents of the report from their saved weights."""

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax

from bomberman.agents import A2CObjective, PolicyAgent
from bomberman.agents.alphazero import AlphaZeroAgent
from bomberman.agents.networks import KartalCNNActorCriticNetwork
from bomberman.game import Game, Settings
from bomberman.training import (
    CoinPickupReward,
    DeathPenalty,
    KillReward,
    PBRSReward,
    StepPenalty,
    UrgentCrateReward,
)


def load_network(path: str | Path, settings: Settings) -> KartalCNNActorCriticNetwork:
    """Load the FC128 (Kartal) network from ``path``."""
    template = KartalCNNActorCriticNetwork.for_settings(settings, jax.random.key(0))
    return eqx.tree_deserialise_leaves(path, template)


def versus_reward(settings: Settings):
    """The reward of the last curriculum stage. The critic learned to predict this reward."""
    return (CoinPickupReward() + KillReward() + DeathPenalty(death=3.0) + StepPenalty(step=0.02)
            + UrgentCrateReward(urgent_crate_scale=1.0) + PBRSReward(settings))


def a2c_agent(network: KartalCNNActorCriticNetwork) -> PolicyAgent:
    """The A2C agent, which plays the network's policy."""
    return PolicyAgent(network, A2CObjective())


def mcts_agent(network: KartalCNNActorCriticNetwork, settings: Settings, num_simulations: int = 16) -> AlphaZeroAgent:
    """The MCTS agent, which uses the same network to guide a search in our engine."""
    return AlphaZeroAgent(network=network, game=Game(settings=settings), reward_fn=versus_reward(settings),
                          num_simulations=num_simulations)
