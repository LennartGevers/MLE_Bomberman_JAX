"""Our MCTS agent from section 6.4 of the report, packaged for the original bomberman_rl game.

To use it, copy this folder into ``bomberman_rl/agent_code/`` and install this repository into
the same Python environment (``pip install -e <path to this repository>``). Then run, for example,
``python main.py play --agents mcts_kartal rule_based_agent``.

The agent plans ahead by simulating the game in our JAX engine. ``NUM_SIMULATIONS`` controls how
much it searches per move. On a slow machine, make sure a move still takes less than 0.5 s.
"""

from pathlib import Path

import equinox as eqx
import jax

import settings as s  # settings.py of the original engine
from bomberman.game import Game
from bomberman.interop import ACTION_NAMES, RoundTracker, classic_settings, legal_mask
from bomberman.trained import load_network, mcts_agent

WEIGHTS = Path(__file__).parent / "weights.eqx"
NUM_SIMULATIONS = 16


@eqx.filter_jit
def _decide(agent, obs, key):
    return agent.greedy_action(obs, legal_mask(obs), key)[0]


def setup(self):
    settings = classic_settings(size=s.COLS, agents=s.MAX_AGENTS, max_steps=s.MAX_STEPS,
                                coins=s.SCENARIOS["classic"]["COIN_COUNT"])
    self.agent = mcts_agent(load_network(WEIGHTS, settings), settings, num_simulations=NUM_SIMULATIONS)
    self.tracker = RoundTracker(settings)
    # compile here, so the first act() call is not slowed down by it
    game = Game(settings=settings)
    _decide(self.agent, game.observe(game.init(jax.random.key(0)), 0), jax.random.key(0))


def act(self, game_state: dict) -> str:
    obs = self.tracker.observe(game_state)
    action = ACTION_NAMES[int(_decide(self.agent, obs, jax.random.key(game_state["step"])))]
    self.tracker.after_action(action)
    return action
