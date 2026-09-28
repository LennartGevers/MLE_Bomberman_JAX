"""Line-ups: which agent plays in which seat."""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx

from bomberman.agents.base import Agent


class Seats(eqx.Module):
    """One agent playing ``count`` seats with the same weights."""

    agent: Agent
    count: int = eqx.field(static=True)

    def __check_init__(self):
        if self.count < 1:
            raise ValueError(f"Seats needs a count of at least 1, got {self.count}")


def unpack_lineup(
    lineup: Sequence[Agent | Seats],
) -> tuple[tuple[Agent, ...], tuple[int, ...]]:
    """Split a line-up into the distinct agents and the agent index of every seat."""
    agents, seat_agent = [], []
    for entry in lineup:
        seats = entry if isinstance(entry, Seats) else Seats(entry, 1)
        if not isinstance(seats.agent, Agent):
            raise TypeError(
                "line-up entries must be an Agent or Seats(agent, count), "
                f"got {type(entry).__name__}"
            )
        seat_agent.extend([len(agents)] * seats.count)
        agents.append(seats.agent)
    return tuple(agents), tuple(seat_agent)


def seats_of(seat_agent: Sequence[int], agent: int) -> tuple[int, ...]:
    """The seats played by ``agent``."""
    return tuple(seat for seat, owner in enumerate(seat_agent) if owner == agent)
