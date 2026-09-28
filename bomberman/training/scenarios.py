"""Reward functions and the boards they are meant for.

Small rewards can be combined with ``+`` and ``*``, as in ``CoinReward``.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from bomberman.game import GameRules, Scenario, Settings

from .base import RewardFn
from .signals import (
    alive,
    coins_collected,
    crates_destroyed,
    died,
    loose_coin,
    pbrs_delta,
)


class NoReward(RewardFn):
    def __call__(self, prev_state, actions, next_state):
        return jnp.zeros_like(coins_collected(prev_state, next_state))
class CoinPickupReward(RewardFn):
    """``coin`` for every collected coin."""

    coin: float = 1.0

    def __call__(self, prev_state, actions, next_state):
        return self.coin * coins_collected(prev_state, next_state)


class StepPenalty(RewardFn):
    """``-step`` for every tick the agent is alive."""

    step: float = 0.02

    def __call__(self, prev_state, actions, next_state):
        return -self.step * alive(next_state)


class DeathPenalty(RewardFn):
    """``-death`` when the agent dies."""

    death: float = 2.0

    def __call__(self, prev_state, actions, next_state):
        return -self.death * died(prev_state, next_state)


class KillReward(RewardFn):
    """``kill`` for every other agent that dies this tick."""

    kill: float = 2.5

    def __call__(self, prev_state, actions, next_state):
        deaths = died(prev_state, next_state)
        rivals_down = deaths.sum() - deaths
        return self.kill * rivals_down


class UrgentCrateReward(RewardFn):
    """``crate`` for every destroyed crate.

    Multiplied by ``urgent_crate_scale`` while a visible coin is still on the board.
    """

    crate: float = 0.3
    urgent_crate_scale: float = 0.25

    def __call__(self, prev_state, actions, next_state):
        urgent = loose_coin(prev_state)
        crate_w = self.crate * jnp.where(urgent, self.urgent_crate_scale, 1.0)
        return crate_w * crates_destroyed(prev_state, next_state) * alive(next_state)


class UrgentStepPenalty(RewardFn):
    """``-step`` for every tick alive, and ``-urgent_step`` more while a visible coin is not collected."""

    step: float = 0.01
    urgent_step: float = 0.05

    def __call__(self, prev_state, actions, next_state):
        urgent = loose_coin(prev_state)
        step_w = self.step + jnp.where(urgent, self.urgent_step, 0.0)
        return -step_w * alive(next_state)


class PBRSReward(RewardFn):
    """Potential-based reward shaping: pulls agents towards coins and away from explosions.

    It does not change which policy is optimal, so it can also be used in the last stage.
    """

    settings: Settings
    enabled: bool = eqx.field(static=True, default=True)
    scale: float = 1.0
    gamma: float = 0.99
    danger_weight: float = 1.0
    coin_weight: float = 1.0

    def __call__(self, prev_state, actions, next_state):
        if not self.enabled:
            return jnp.zeros_like(prev_state.alive, dtype=jnp.float32)
        delta = pbrs_delta(
            prev_state,
            next_state,
            self.settings,
            gamma=self.gamma,
            danger_weight=self.danger_weight,
            coin_weight=self.coin_weight,
        )
        return self.scale * delta


def CoinReward(*, coin: float = 1.0, step: float = 0.02) -> RewardFn:
    """Reward for collecting visible coins fast."""
    return CoinPickupReward(coin=coin) + StepPenalty(step=step)


def CrateReward(
    *,
    coin: float = 1.0,
    crate: float = 0.3,
    death: float = 2.0,
    step: float = 0.01,
    urgent_step: float = 0.05,
    urgent_crate_scale: float = 0.25,
) -> RewardFn:
    """Reward for bombing crates to find coins without dying."""
    return (
        CoinPickupReward(coin=coin)
        + UrgentCrateReward(crate=crate, urgent_crate_scale=urgent_crate_scale)
        + DeathPenalty(death=death)
        + UrgentStepPenalty(step=step, urgent_step=urgent_step)
    )


def VersusReward(
    *, coin: float = 1.0, kill: float = 2.5, death: float = 3.0, step: float = 0.02
) -> RewardFn:
    """Reward for playing against others: coins, kills, and penalties for dying and for every step."""
    return (
        CoinPickupReward(coin=coin)
        + KillReward(kill=kill)
        + DeathPenalty(death=death)
        + StepPenalty(step=step)
    )


def coin_task_settings(
    *,
    size: int = 9,
    coins: int = 8,
    max_steps: int = 128,
    crate_density: float = 0.0,
    agents: int = 1,
) -> Settings:
    """Settings for a board with coins and no crates."""
    return Settings(
        scenario=Scenario(
            crate_density=crate_density, coin_count=coins, size=size, agent_count=agents
        ),
        rules=GameRules(max_steps=max_steps),
    )


def crate_task_settings(
    *,
    size: int = 9,
    coins: int = 8,
    crate_density: float = 0.25,
    max_steps: int = 160,
    agents: int = 1,
) -> Settings:
    """Settings for a board where coins are under crates."""
    return coin_task_settings(
        size=size,
        coins=coins,
        max_steps=max_steps,
        crate_density=crate_density,
        agents=agents,
    )


def versus_task_settings(
    *,
    size: int = 11,
    coins: int = 9,
    crate_density: float = 0.35,
    max_steps: int = 200,
    agents: int = 2,
) -> Settings:
    """Settings for a board with crates and ``agents`` players."""
    return crate_task_settings(
        size=size,
        coins=coins,
        crate_density=crate_density,
        max_steps=max_steps,
        agents=agents,
    )
