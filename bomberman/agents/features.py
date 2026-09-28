"""Turn an observation into the input of a network.

``ChannelStack`` is the default. ``RichChannelStack`` adds five more planes.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from bomberman.game import Game, GameRules, Observation, Settings

# Index of each plane built by `ChannelStack`. The scripted agents also use them.
# `FIELD_CH` is not called `FIELD` because that name is already taken by the
# tile types in `bomberman.game`.
FIELD_CH, AGENTS, COINS, COOLDOWN, FIRE, BOMBS = range(6)

# The planes that `RichChannelStack` adds.
LIVE_AGENTS, OPPONENT_COOLDOWN, OWN_BOMB, SCORE_LEAD, TIME_LEFT = range(6, 11)


class FeatureShaping(eqx.Module):
    """Turns an ``Observation`` into the input array of a network."""

    def __call__(self, obs: Observation) -> Array:
        """The ``(C, H, W)`` input for ``obs``."""
        raise NotImplementedError

    def shape(self, settings: Settings | Game) -> tuple[int, ...]:
        """The shape of the output for a board with ``settings``."""
        game = settings if isinstance(settings, Game) else Game(settings=settings)
        spec = jax.eval_shape(
            lambda key: self(game.observe(game.init(key), jnp.int32(0))),
            jax.random.PRNGKey(0),
        )
        return tuple(spec.shape)


class ChannelStack(FeatureShaping):
    """Six planes: field, agents, coins, cooldown, explosions and bombs."""

    def __call__(self, obs: Observation) -> Array:
        board = obs.board
        field_map = board.field_map
        me = obs.position

        agents_map = (
            jnp.zeros_like(field_map)
            .at[board.agent_positions[:, 0], board.agent_positions[:, 1]]
            .set(1)
            .at[me[0], me[1]]
            .set(-1)
        )
        cooldown_map = obs.cooldown * jnp.ones_like(field_map)

        return jnp.stack(
            [
                field_map,
                agents_map,
                board.coins.map(field_map),
                cooldown_map,
                board.explosions.map(field_map),
                board.bombs.map(field_map),
            ],
            axis=0,
        )


class RichChannelStack(ChannelStack):
    """``ChannelStack`` plus five planes, eleven in total.

    The extra planes show which opponents are alive, their cooldowns, our own bomb, our score lead and the time left.
    """

    # Taken from the default `GameRules`. For other rules use `for_settings`,
    # otherwise the time-left plane is scaled wrongly.
    max_steps: int = eqx.field(static=True, default=GameRules().max_steps)
    score_scale: float = eqx.field(static=True, default=GameRules().reward_kill)

    def __check_init__(self):
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if self.score_scale <= 0.0:
            raise ValueError("score_scale must be > 0")

    @classmethod
    def for_settings(cls, settings: Settings | Game, **kwargs) -> RichChannelStack:
        """Build a stack that uses ``max_steps`` from ``settings``."""
        rules = (settings.settings if isinstance(settings, Game) else settings).rules
        return cls(max_steps=rules.max_steps, **kwargs)

    def __call__(self, obs: Observation) -> Array:
        board = obs.board
        field_map = board.field_map
        positions = board.agent_positions
        blank = jnp.zeros_like(field_map, jnp.float32)

        # Living opponents only. Our own position is already in plane 1.
        seats = jnp.arange(obs.num_agents)
        other = (seats != obs.agent_id) & obs.alive
        scatter = lambda values: blank.at[positions[:, 0], positions[:, 1]].max(
            jnp.where(other, values, 0.0).astype(jnp.float32)
        )

        bombs = board.bombs
        mine = obs.agent_id  # the slot of a bomb or explosion is its owner
        own_bomb = blank.at[bombs.pos[mine, 0], bombs.pos[mine, 1]].max(
            jnp.where(bombs.active[mine], bombs.timer[mine], 0).astype(jnp.float32)
        )

        # Lead over the best opponent. Our own score is replaced by the lowest score
        # so it does not count as an opponent; with no opponents the lead is 0.
        rivals = jnp.where(seats != obs.agent_id, obs.score, obs.score.min())
        lead = obs.score[obs.agent_id] - rivals.max()

        progress = jnp.clip(
            (self.max_steps - obs.step_count.astype(jnp.float32)) / self.max_steps,
            0.0,
            1.0,
        )

        return jnp.concatenate(
            [
                super().__call__(obs).astype(jnp.float32),
                jnp.stack(
                    [
                        scatter(jnp.ones_like(obs.alive, jnp.float32)),
                        scatter(obs.bomb_cooldown + 1),
                        own_bomb,
                        jnp.tanh(lead / self.score_scale) * jnp.ones_like(blank),
                        progress * jnp.ones_like(blank),
                    ],
                    axis=0,
                ),
            ],
            axis=0,
        )
