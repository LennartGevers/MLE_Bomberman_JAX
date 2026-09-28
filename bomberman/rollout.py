"""``RolloutOutput``: everything recorded while playing a batch of games."""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from bomberman.game import GameState, Observation


class RolloutOutput(eqx.Module):
    """The record of ``E`` games played in parallel for up to ``R`` ticks.

    Most fields have shape ``(E, R, ...)``. ``states`` also includes the starting state, so it has ``R + 1`` ticks.
    """

    states: GameState      # (E, R+1, ...) game state at ticks 0 to R
    obs: Observation       # (E, R, seats) what each agent saw
    legal_mask: Array      # (E, R, seats, N_ACTIONS) bool
    action: Array          # (E, R, seats) int32
    key: PRNGKeyArray      # (E, R, seats) random key passed to the agent's `act`
    done: Array            # (E, R) bool, True once the game is over
    reward: Array          # (E, R, seats) float32
    coins: Array           # (E, seats) float32, coins collected in the whole game
    alive: Array           # (E, seats) bool, alive at the end
    steps: Array           # (E,) float32, number of ticks played

    def frames(self, env: int = 0) -> list[GameState]:
        """The states of game ``env``, up to the tick it ended."""
        length = int(self.steps[env]) + 1
        episode = jax.tree_util.tree_map(lambda a: a[env, :length], self.states)
        return [jax.tree_util.tree_map(lambda a, t=t: a[t], episode) for t in range(length)]

    def outcome_metrics(self, seat_agent: Sequence[int] | None = None) -> dict[str, Array]:
        """Statistics per agent and game."""
        return outcome_metrics(
            coins=self.coins,
            alive=self.alive,
            final_score=self.states.score[:, -1, :],
            steps=self.steps,
            seat_agent=seat_agent,
        )


def outcome_metrics(
    *,
    coins: Array,
    alive: Array,
    final_score: Array,
    steps: Array,
    seat_agent: Sequence[int] | None = None,
) -> dict[str, Array]:
    """Statistics per agent, averaged over all games.

    An agent wins a game only if its final score is higher than everyone else's.
    """
    n_seats = coins.shape[-1]
    seat_agent = tuple(seat_agent) if seat_agent is not None else tuple(range(n_seats))
    num_agents = max(seat_agent) + 1

    is_top_score = final_score == final_score.max(axis=-1, keepdims=True)
    winner = is_top_score & (is_top_score.sum(axis=-1, keepdims=True) == 1)  # (E, seats)

    metrics = {"world_steps": steps.mean()}
    for i in range(num_agents):
        seats = [seat for seat, owner in enumerate(seat_agent) if owner == i]
        metrics[f"agent{i}/coins"] = coins[:, seats].mean(axis=-1).mean()
        metrics[f"agent{i}/death"] = (
            (~alive[:, seats]).astype(jnp.float32).mean(axis=-1).mean()
        )
        if n_seats > 1:
            metrics[f"agent{i}/win"] = (
                winner[:, seats].astype(jnp.float32).mean(axis=-1).mean()
            )
    return metrics
