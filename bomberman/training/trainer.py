"""``Training``: trains agents playing together on one board."""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from diffrax import AbstractProgressMeter, NoProgressMeter
from jaxtyping import Array, PRNGKeyArray, PyTree

from bomberman.agents.base import Agent, Batch
from bomberman.game import Game, Settings
from bomberman.rollout import RolloutOutput

from .base import RewardFn
from .checkpoint import Checkpointer, NoCheckpointer
from .lineup import Seats, seats_of, unpack_lineup
from .logger import Logger, NoLogger
from .rollout import Rollout
from .scenarios import CoinReward


def _seat_columns(tree: PyTree, seats: Sequence[int]) -> PyTree:
    """Stack the records of the given seats along the first axis."""
    return jax.tree_util.tree_map(
        lambda a: jnp.concatenate([a[:, :, s] for s in seats]), tree
    )


def _batches(output: RolloutOutput, seat_agent, progress, opt_states) -> list[Batch]:
    """Split a rollout into one ``Batch`` per agent."""
    batches = []
    for i, opt_state in enumerate(opt_states):
        seats = seats_of(seat_agent, i)
        batches.append(
            Batch(
                obs=_seat_columns(output.obs, seats),
                legal_mask=_seat_columns(output.legal_mask, seats),
                action=_seat_columns(output.action, seats),
                reward=_seat_columns(output.reward, seats),
                mask=jnp.concatenate([~output.done] * len(seats)),
                key=_seat_columns(output.key, seats),
                progress=progress,
                opt_state=opt_state,
            )
        )
    return batches


def _is_trainable(agent: Agent) -> bool:
    return len(jax.tree_util.tree_leaves(eqx.filter(agent, eqx.is_inexact_array))) > 0


def _update_agents(agents, opt_states, batches, optimisers):
    """One gradient step for every agent that learns."""
    new_agents, new_states, metrics = [], [], {}
    for i, (agent, opt, opt_state, batch) in enumerate(
        zip(agents, optimisers, opt_states, batches)
    ):
        if opt is None:
            new_agents.append(agent)
            new_states.append(opt_state)
            continue
        (_, aux), grads = eqx.filter_value_and_grad(
            lambda a, b: a.loss(b), has_aux=True
        )(agent, batch)
        updates, opt_state = opt.update(grads, opt_state)
        new_agents.append(eqx.apply_updates(agent, updates))
        new_states.append(opt_state)
        metrics.update({f"agent{i}/{k}": v for k, v in aux.items()})
    return tuple(new_agents), tuple(new_states), metrics


class Training:
    """Trains agents that play together on one board.

    Agents that do not learn, like the scripted ones, are simply opponents.
    """

    def __init__(
        self,
        settings: Settings | Game,
        lineup: Sequence[Agent | Seats],
        *,
        optimisers=None,
        reward_fn: RewardFn | None = None,
        num_envs: int = 64,
        forbid_bomb: bool = False,
        logger: Logger | None = None,
        checkpointer: Checkpointer | None = None,
        progress_meter: AbstractProgressMeter | None = None,
        learning_rate: float = 3e-4,
        max_grad_norm: float = 0.5,
    ):
        self.game = settings if isinstance(settings, Game) else Game(settings=settings)
        self.settings = self.game.settings
        self.agents, self.seat_agent = unpack_lineup(lineup)

        seats = len(self.seat_agent)
        expected = self.settings.scenario.agent_count
        if seats != expected:
            raise ValueError(
                f"settings seat {expected} players but the line-up fills {seats} seats"
            )
        if not 1 <= seats <= 4:
            raise ValueError("Bomberman seats one to four players")

        self.reward_fn = reward_fn if reward_fn is not None else CoinReward()
        self.num_envs = int(num_envs)
        self.rounds = int(self.settings.rules.max_steps)
        self.rollout = Rollout(
            self.game, self.reward_fn, self.rounds, forbid_bomb=forbid_bomb
        )
        self.logger = logger or NoLogger()
        self.checkpointer = checkpointer or NoCheckpointer()
        self.progress_meter = progress_meter or NoProgressMeter()
        self.optimisers = self._resolve_optimisers(
            optimisers, learning_rate, max_grad_norm
        )
        self.opt_states = tuple(
            opt.init(eqx.filter(a, eqx.is_inexact_array)) if opt is not None else None
            for a, opt in zip(self.agents, self.optimisers)
        )
        self.history: dict[str, Array] = {}

    def _resolve_optimisers(self, optimisers, learning_rate, max_grad_norm):
        if optimisers is None:
            optimisers = optax.chain(
                optax.clip_by_global_norm(max_grad_norm), optax.adam(learning_rate)
            )
        # An Optax optimiser is itself a tuple, so check for one before calling `tuple()`.
        if isinstance(
            optimisers,
            (optax.GradientTransformation, optax.GradientTransformationExtraArgs),
        ):
            return tuple(optimisers if _is_trainable(a) else None for a in self.agents)
        optimisers = tuple(optimisers)
        if len(optimisers) != len(self.agents):
            raise ValueError("pass one optimiser per line-up entry, or a single shared one")
        return optimisers

    def _seats(self, agents: Sequence[Agent]) -> list[Agent]:
        """One agent per seat."""
        return [agents[i] for i in self.seat_agent]

    def fit(
        self,
        key: PRNGKeyArray,
        iterations: int,
        *,
        logger: Logger | None = None,
        checkpointer: Checkpointer | None = None,
    ) -> tuple[Agent, ...]:
        """Train for ``iterations`` updates and return the trained agents."""
        logger = logger or self.logger
        checkpointer = checkpointer or self.checkpointer
        # Skip the callbacks entirely when nothing is logged or saved.
        log = not isinstance(logger, NoLogger)
        checkpoint = not isinstance(checkpointer, NoCheckpointer)
        meter = self.progress_meter
        params, static = eqx.partition(self.agents, eqx.is_inexact_array)
        span = max(iterations - 1, 1)

        @eqx.filter_jit
        def run(params, opt_states, meter_state, key):
            def body(carry, i):
                params, opt_states, meter_state, key = carry
                agents = eqx.combine(params, static)
                progress = i.astype(jnp.float32) / span
                meter_state = meter.step(meter_state, progress)

                key, rollout_key = jax.random.split(key)
                output = self.rollout(
                    self._seats(agents), rollout_key, greedy=False, num_envs=self.num_envs
                )
                batches = _batches(output, self.seat_agent, progress, opt_states)
                agents, opt_states, losses = _update_agents(
                    agents, opt_states, batches, self.optimisers
                )
                metrics = {
                    "progress": progress,
                    **output.outcome_metrics(self.seat_agent),
                    **losses,
                }

                if log:
                    jax.lax.cond(
                        i % logger.every == 0,
                        lambda m: jax.debug.callback(logger, i, m),
                        lambda m: None,
                        metrics,
                    )
                if checkpoint:
                    # The agents are not passed as a `cond` operand because they can contain
                    # functions such as `jax.nn.relu`. `ordered=True` makes sure no
                    # checkpoint is skipped or saved out of order.
                    jax.lax.cond(
                        i % checkpointer.every == 0,
                        lambda: jax.debug.callback(
                            checkpointer, i, agents, opt_states, ordered=True
                        ),
                        lambda: None,
                    )

                params = eqx.filter(agents, eqx.is_inexact_array)
                return (params, opt_states, meter_state, key), metrics

            return jax.lax.scan(
                body,
                (params, opt_states, meter_state, key),
                jnp.arange(iterations, dtype=jnp.int32),
            )

        (params, opt_states, meter_state, _), history = run(
            params, self.opt_states, meter.init(), key
        )
        meter.close(meter_state)
        checkpointer.wait()  # wait for the last save to finish
        self.agents = eqx.combine(params, static)
        self.opt_states = opt_states
        self.history = history
        return self.agents

    def evaluate(
        self, key: PRNGKeyArray, episodes: int = 128, *, greedy: bool = True
    ) -> dict[str, float]:
        """Play ``episodes`` games without learning and return statistics per agent."""
        metrics = eqx.filter_jit(
            lambda agents, k: self.rollout(
                self._seats(agents), k, greedy=greedy, num_envs=int(episodes)
            ).outcome_metrics(self.seat_agent)
        )(self.agents, key)
        return {k: float(v) for k, v in metrics.items()}
