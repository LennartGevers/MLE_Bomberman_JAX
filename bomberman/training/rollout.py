"""``Rollout`` plays many games in parallel and records them."""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from bomberman.agents.base import Agent
from bomberman.game import ACTION, Game
from bomberman.rollout import RolloutOutput

from .base import RewardFn
from .signals import coins_collected

N_ACTIONS = len(ACTION)


def _row(tree, i):
    """Row ``i`` of every array in a pytree."""
    return jax.tree_util.tree_map(lambda a: a[i], tree)


def action_mask(forbid_bomb: bool) -> Array:
    """All actions, without BOMB if ``forbid_bomb`` is set."""
    mask = jnp.ones(N_ACTIONS, jnp.bool_)
    if forbid_bomb:
        mask = mask.at[ACTION.BOMB].set(False)
    return mask


class Rollout:
    """Plays games with one agent per seat and records rewards."""

    def __init__(
        self, game: Game, reward_fn: RewardFn, rounds: int, *, forbid_bomb: bool = False
    ):
        self.game = game
        self.reward_fn = reward_fn
        self.rounds = int(rounds)
        self.action_mask = action_mask(forbid_bomb)

    def __call__(
        self,
        agents: Sequence[Agent],
        key: PRNGKeyArray,
        *,
        greedy: bool = False,
        num_envs: int,
    ) -> RolloutOutput:
        """Play ``num_envs`` games in parallel, with ``agents[s]`` in seat ``s``."""
        return jax.vmap(lambda k: self._episode(agents, k, greedy=greedy))(
            jax.random.split(key, num_envs)
        )

    def _episode(
        self, agents: Sequence[Agent], key: PRNGKeyArray, *, greedy: bool
    ) -> RolloutOutput:
        """Play one game and record every tick."""
        game, reward_fn = self.game, self.reward_fn
        n = len(agents)
        seats = jnp.arange(n)
        agent_states0 = [agent.init() for agent in agents]
        init_key, scan_key = jax.random.split(key)
        state0 = game.init(init_key)

        def tick(carry, tick_key):
            state, done, agent_states = carry
            obs = jax.vmap(lambda s: game.observe(state, s))(seats)
            legal = (
                jax.vmap(lambda s: game.legal_action_mask(state, s))(seats)
                & self.action_mask
            )
            keys = jax.random.split(tick_key, n)

            actions, new_agent_states = [], []
            for s, agent, agent_state in zip(range(n), agents, agent_states):
                choose = agent.greedy_action if greedy else agent.act
                chosen, _value, agent_state = choose(
                    _row(obs, s), legal[s], keys[s], agent_state
                )
                actions.append(chosen.astype(jnp.int32))
                new_agent_states.append(agent_state)
            actions = jnp.stack(actions)

            nxt = jax.lax.cond(
                done, lambda s, a: s, lambda s, a: game.step(s, a), state, actions
            )
            reward = jnp.where(done, 0.0, reward_fn(state, actions, nxt))
            coins = jnp.where(done, 0.0, coins_collected(state, nxt))
            outputs = (nxt, obs, legal, actions, keys, done, reward, coins)
            return (nxt, done | game.is_terminal(nxt), new_agent_states), outputs

        carry0 = (state0, jnp.bool_(False), agent_states0)
        (final, _, _), (states, obs, legal, action, key, done, reward, coins) = (
            jax.lax.scan(tick, carry0, jax.random.split(scan_key, self.rounds))
        )
        states = jax.tree_util.tree_map(
            lambda s0, s: jnp.concatenate([s0[None], s], axis=0), state0, states
        )
        return RolloutOutput(
            states=states,
            obs=obs,
            legal_mask=legal,
            action=action,
            key=key,
            done=done,
            reward=reward,
            coins=coins.sum(axis=0),
            alive=final.alive,
            steps=final.step_count.astype(jnp.float32),
        )
