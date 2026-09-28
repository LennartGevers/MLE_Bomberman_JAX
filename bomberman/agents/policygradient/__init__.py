"""Policy-gradient agents: a network trained with an objective such as REINFORCE or A2C."""

from bomberman.agents.policygradient.a2c import A2CObjective
from bomberman.agents.policygradient.agent import PolicyAgent
from bomberman.agents.policygradient.objective import Objective
from bomberman.agents.policygradient.reinforce import ReinforceObjective

__all__ = [
    "A2CObjective",
    "Objective",
    "PolicyAgent",
    "ReinforceObjective",
]
