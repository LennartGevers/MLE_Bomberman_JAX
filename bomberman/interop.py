"""Lets our agents play in the original ``bomberman_rl`` game.

The original game hands an agent a ``game_state`` dictionary; this module turns it into the
``Observation`` our agents expect. The board uses the same numbers in both games (``-1`` wall,
``0`` empty, ``1`` crate) and moves work the same way (``UP`` is ``y - 1``). Some information is
missing from ``game_state`` and has to be filled in:

* **Timers.** Bomb and explosion timers in the original game are one lower than ours, so we add one.
* **Cooldowns.** The original game only says whether an agent can plant a bomb. We count our own
  cooldown from our own actions and estimate the others' from their bombs on the board.
* **Coins and bomb owners.** The original game only lists visible coins, and bombs have no owner.
  A coin we saw earlier that is now gone counts as collected, the remaining coins are assumed to be
  under crates, and each bomb is given to an agent that currently cannot plant one.

All arrays are padded to a fixed size (four agents, ``coin_count`` coins), so the compiled agent
sees the same shapes on every call and is never recompiled during a game.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from bomberman.game import ACTION, GameRules, GameState, Observation, Scenario, Settings
from bomberman.game.dynamics import legal_action_mask
from bomberman.game.environment import HIDDEN_COIN, Bombs, Coins, EnvironmentState, Explosions, blast_width

ACTION_NAMES = [action.name for action in ACTION]  # UP, DOWN, LEFT, RIGHT, BOMB, WAIT


def classic_settings(size: int = 17, agents: int = 4, coins: int = 9, max_steps: int = 400) -> Settings:
    """Our game settings matching the classic mode of the original game."""
    return Settings(
        scenario=Scenario(crate_density=0.75, coin_count=coins, size=size, agent_count=agents),
        rules=GameRules(max_steps=max_steps),
    )


class RoundTracker:
    """Remembers what we need between two ``act`` calls of the same round."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.start_round(None)

    def start_round(self, round_id) -> None:
        self.round, self.cooldown, self.seen_coins = round_id, 0, []
        self.position, self.bomb_position = None, None

    def after_action(self, action: str) -> None:
        """Update our cooldown and our bomb after we chose ``action``."""
        if action == "BOMB" and self.cooldown == 0:
            self.cooldown = self.settings.rules.bomb_cooldown - 1
            self.bomb_position = self.position
        else:
            self.cooldown = max(self.cooldown - 1, 0)

    def observe(self, game_state: dict) -> Observation:
        """Turn ``game_state`` into an observation. We are always agent ``0``."""
        if game_state["round"] != self.round:
            self.start_round(game_state["round"])
        _, _, bombs_left, self.position = game_state["self"]
        if bombs_left:
            self.cooldown = 0
        for xy in map(tuple, game_state["coins"]):
            if xy not in self.seen_coins:
                self.seen_coins.append(xy)
        own_bomb = tuple(self.bomb_position) if self.cooldown > self.settings.rules.explosion_timer else None
        return observation_from_game_state(game_state, self.cooldown, own_bomb, self.seen_coins, self.settings)


def observation_from_game_state(
    game_state: dict, own_cooldown: int, own_bomb, seen_coins: list, settings: Settings
) -> Observation:
    n = settings.scenario.agent_count
    rules = settings.rules
    field = np.asarray(game_state["field"], dtype=np.int32)

    # Agents: we are agent 0 and the others follow. Missing agents count as dead and sit on a wall tile.
    _, own_score, _, own_pos = game_state["self"]
    others = game_state["others"]
    positions = np.zeros((n, 2), np.int32)
    alive = np.zeros(n, bool)
    score = np.zeros(n, np.float32)
    can_plant = np.zeros(n, bool)
    for seat, (_, points, bombs_left, pos) in enumerate([("self", own_score, True, own_pos), *others]):
        positions[seat], alive[seat], score[seat], can_plant[seat] = pos, True, points, bombs_left

    # Coins: visible coins as listed, coins we saw earlier as collected, the rest hidden.
    visible = set(map(tuple, game_state["coins"]))
    coin_count = settings.scenario.coin_count
    coin_pos = np.tile(np.asarray(HIDDEN_COIN, np.int32), (coin_count, 1))
    collected = np.zeros(coin_count, bool)
    for i, xy in enumerate(seen_coins[:coin_count]):
        coin_pos[i], collected[i] = xy, xy not in visible

    # Bombs: ours goes to slot 0, every other bomb to an agent that currently cannot plant.
    bomb_pos = np.zeros((n, 2), np.int32)
    bomb_timer = np.zeros(n, np.int32)
    bomb_active = np.zeros(n, bool)
    cooldown = np.zeros(n, np.int32)
    cooldown[0] = own_cooldown
    free_slots = [seat for seat in range(1, n) if alive[seat] and not can_plant[seat]]
    free_slots += [seat for seat in range(1, n) if seat not in free_slots]
    for xy, t in game_state["bombs"]:
        xy = tuple(xy)
        seat = 0 if xy == own_bomb and not bomb_active[0] else free_slots.pop(0)
        bomb_pos[seat], bomb_timer[seat], bomb_active[seat] = xy, t + 1, True
        if seat != 0:
            cooldown[seat] = t + 1 + rules.explosion_timer
    for seat in range(1, n):
        if alive[seat] and not can_plant[seat] and not bomb_active[seat]:
            cooldown[seat] = 1  # its bomb has exploded, but the explosion is still there

    # Explosions: group the burning tiles by timer and store each group in one row.
    width = blast_width(rules.bomb_power)
    exp_map = np.asarray(game_state["explosion_map"])
    blast = np.zeros((n, width, 2), np.int32)
    exp_timer = np.zeros(n, np.int32)
    exp_active = np.zeros(n, bool)
    row = 0
    for timer in sorted({int(v) for v in exp_map[exp_map > 0]}, reverse=True):
        tiles = np.argwhere(exp_map == timer)
        for start in range(0, len(tiles), width):
            if row == n:
                break
            chunk = tiles[start : start + width]
            blast[row] = chunk[np.arange(width) % len(chunk)]  # fill up the row by repeating tiles
            exp_timer[row], exp_active[row] = timer + 1, True
            row += 1

    board = EnvironmentState(
        field_map=jnp.asarray(field),
        agent_positions=jnp.asarray(positions),
        coins=Coins(pos=jnp.asarray(coin_pos), collected=jnp.asarray(collected)),
        bombs=Bombs(pos=jnp.asarray(bomb_pos), timer=jnp.asarray(bomb_timer), active=jnp.asarray(bomb_active)),
        explosions=Explosions(blast=jnp.asarray(blast), timer=jnp.asarray(exp_timer), active=jnp.asarray(exp_active)),
    )
    return Observation(
        board=board,
        agent_id=jnp.int32(0),
        bomb_cooldown=jnp.asarray(cooldown),
        alive=jnp.asarray(alive),
        score=jnp.asarray(score),
        step_count=jnp.int32(max(int(game_state["step"]) - 1, 0)),
    )


def legal_mask(obs: Observation) -> jax.Array:
    """Our legal actions under the rules of our engine."""
    state = GameState(
        board=obs.board,
        turn_order=jnp.arange(obs.num_agents),
        bomb_cooldown=obs.bomb_cooldown,
        alive=obs.alive,
        score=obs.score,
        _key=jax.random.key(0),
        step_count=obs.step_count,
    )
    return legal_action_mask(state, jnp.int32(0))
