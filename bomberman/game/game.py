"""``Game``, the entry point to the engine, and ``GameState``."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Key

from bomberman.game.dynamics import is_terminal, legal_action_mask, step
from bomberman.game.environment import EnvironmentState, init_environment
from bomberman.game.observe import Observation, observe
from bomberman.game.settings import Settings


class GameState(eqx.Module):
    board: EnvironmentState
    turn_order: Array          # (num_agents,) the order in which actions are carried out this tick (secret)
    bomb_cooldown: Array       # (num_agents,) ticks until each agent may plant again
    alive: Array               # (num_agents,) bool
    score: Array               # (num_agents,) game score so far (coins and kills)
    _key: Key
    step_count: Array = eqx.field(default_factory=lambda: jnp.int32(0))

    @property
    def num_agents(self) -> int:
        return self.turn_order.shape[0]

    @property
    def key(self) -> Key:
        return jax.random.fold_in(self._key, self.step_count)


class Game(eqx.Module):
    settings: Settings = Settings()

    def init(self, key: Array) -> GameState:
        env_key, order_key, step_key = jax.random.split(key, 3)
        n = self.settings.scenario.agent_count

        return GameState(
            board=init_environment(self.settings, env_key),
            turn_order=jax.random.permutation(order_key, jnp.arange(n)),
            bomb_cooldown=jnp.zeros(n, dtype=jnp.int32),
            alive=jnp.ones(n, dtype=jnp.bool_),
            score=jnp.zeros(n, dtype=jnp.float32),
            _key=step_key,
        )

    def step(self, state: GameState, actions: Array) -> GameState:
        """Play one tick, given one action per agent."""
        return step(state, actions, self.settings)

    def observe(self, state: GameState, agent_id: Array) -> Observation:
        """The game as agent ``agent_id`` sees it."""
        return observe(state, agent_id)

    def legal_action_mask(self, state: GameState, agent_id: Array) -> Array:
        """Boolean mask of the actions ``agent_id`` may take this tick."""
        return legal_action_mask(state, agent_id)

    def is_terminal(self, state: GameState) -> Array:
        return is_terminal(state, self.settings)

    def score(self, state: GameState) -> Array:
        return state.score
