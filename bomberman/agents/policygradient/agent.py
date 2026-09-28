"""``PolicyAgent``: an agent made of a network and a training objective."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from bomberman.agents.base import Agent, Batch
from bomberman.agents.features import ChannelStack, FeatureShaping
from bomberman.agents.networks import Network
from bomberman.game import Observation

from .objective import Objective, _masked_log_policy


class PolicyAgent(Agent[None]):
    """Samples its actions from a network and learns with an objective.

    ``features`` turns the observation into the network's input.
    """

    network: Network
    objective: Objective
    features: FeatureShaping = ChannelStack()

    def __check_init__(self):
        if self.objective.needs_value and not self.network.has_value_head:
            raise TypeError(
                f"{type(self.objective).__name__} needs a value estimate from "
                f"the network, but {type(self.network).__name__} has no critic "
                "head (has_value_head=False)"
            )

    def _log_probs_and_value(
        self, obs: Observation, legal_mask: Array, key: PRNGKeyArray, *, inference: bool
    ) -> tuple[Array, Array | None]:
        features = self.features(obs).astype(jnp.float32)
        logits, value = self.network(features, key, inference=inference)
        return _masked_log_policy(logits, legal_mask), value

    def log_policy(
        self,
        obs: Observation,
        legal_mask: Array,
        key: PRNGKeyArray,
        *,
        inference: bool = False,
    ) -> Array:
        """Masked log-probabilities over actions, shape ``(N_ACTIONS,)``."""
        log_probs, _ = self._log_probs_and_value(obs, legal_mask, key, inference=inference)
        return log_probs

    def act(self, obs: Observation, legal_mask: Array, key: PRNGKeyArray, state=None):
        fwd_key, sample_key = jax.random.split(key)
        log_probs, value = self._log_probs_and_value(obs, legal_mask, fwd_key, inference=False)
        action = jax.random.categorical(sample_key, log_probs).astype(jnp.int32)
        return action, value, state

    def greedy_action(
        self, obs: Observation, legal_mask: Array, key: PRNGKeyArray, state=None
    ):
        log_probs, value = self._log_probs_and_value(obs, legal_mask, key, inference=True)
        action = jnp.argmax(log_probs).astype(jnp.int32)
        return action, value, state

    def shaped(self, batch: Batch) -> Batch:
        """``batch`` with its observations turned into network inputs."""
        features = jax.vmap(jax.vmap(self.features))(batch.obs)
        return dataclasses.replace(batch, obs=features)

    def loss(self, batch: Batch) -> tuple[Array, dict[str, Array]]:
        return self.objective(self.network, self.shaped(batch))
