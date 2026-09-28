from enum import IntEnum

import jax.numpy as jnp

TRUE = jnp.bool_(True)


class ACTION(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    BOMB = 4
    WAIT = 5


# (dx, dy) for each move. As in the original game, y grows downwards.
# All other movement tables are built from this one.
DIRECTIONS = {
    ACTION.UP: (0, -1),
    ACTION.DOWN: (0, 1),
    ACTION.LEFT: (-1, 0),
    ACTION.RIGHT: (1, 0),
}

# The same table as an array, so it can be indexed with a traced action id.
# BOMB and WAIT do not move the agent.
DIRECTION_ARR = jnp.asarray(
    [DIRECTIONS.get(action, (0, 0)) for action in ACTION], dtype=jnp.int32
)

# Which branch of `jax.lax.switch` in dynamics._perform_action handles an action:
# 0 moves, 1 plants a bomb, 2 waits.
SWITCH_BRANCH = jnp.asarray([0, 0, 0, 0, 1, 2], dtype=jnp.int32)
