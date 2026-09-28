import equinox as eqx


class Scenario(eqx.Module):
    crate_density: float = 0.75
    coin_count: int = 9
    size: int = 17
    agent_count: int = 4


class GameRules(eqx.Module):
    max_steps: int = 512
    bomb_power: int = 3
    bomb_timer: int = 4
    explosion_timer: int = 2
    reward_coin: float = 1.0
    reward_kill: float = 5.0

    @property
    def bomb_cooldown(self) -> int:
        """Ticks after planting a bomb before the agent can plant the next one."""
        return self.bomb_timer + self.explosion_timer + 1


class Settings(eqx.Module):
    scenario: Scenario = Scenario()
    rules: GameRules = GameRules()
