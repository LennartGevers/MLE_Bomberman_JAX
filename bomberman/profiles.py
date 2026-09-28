"""Training settings for CPU, GPU and TPU."""

from __future__ import annotations

import jax

# num_envs:      games played in parallel for each update
# iterations:    number of updates in a full training run
# eval_episodes: games played in Training.evaluate
# log_every:     print progress every this many updates
DEVICE_PROFILES: dict[str, dict[str, int]] = {
    "cpu": {"num_envs": 48, "iterations": 150, "eval_episodes": 128, "log_every": 25},
    "gpu": {"num_envs": 256, "iterations": 800, "eval_episodes": 512, "log_every": 50},
}
DEVICE_PROFILES["tpu"] = DEVICE_PROFILES["gpu"]


def device_profile(backend: str | None = None) -> dict[str, int]:
    """Training settings for ``backend`` (by default the backend JAX is using)."""
    backend = backend or jax.default_backend()
    return dict(DEVICE_PROFILES.get(backend, DEVICE_PROFILES["cpu"]))


# In addition to the settings of `device_profile`:
# num_simulations:  search steps per move
# determinizations: guesses of where the hidden coins are, per move
# batch_size:       samples per gradient step (Config.batch_size)
#
# The CPU settings finish on a laptop in well under an hour. They only show that
# training runs and will not produce a strong agent. The GPU settings are a first
# guess, not tuned; time one iteration before starting a long run.
ALPHAZERO_PROFILES: dict[str, dict[str, int]] = {
    "cpu": {
        "num_envs": 8,
        "iterations": 150,
        "num_simulations": 24,
        "determinizations": 1,
        "batch_size": 256,
        "eval_episodes": 32,
        "log_every": 10,
    },
    "gpu": {
        "num_envs": 64,
        "iterations": 200,
        "num_simulations": 48,
        "determinizations": 1,
        "batch_size": 1024,
        "eval_episodes": 128,
        "log_every": 10,
    },
}
ALPHAZERO_PROFILES["tpu"] = ALPHAZERO_PROFILES["gpu"]


def alphazero_profile(backend: str | None = None) -> dict[str, int]:
    """AlphaZero training settings for ``backend`` (by default the backend JAX is using)."""
    backend = backend or jax.default_backend()
    return dict(ALPHAZERO_PROFILES.get(backend, ALPHAZERO_PROFILES["cpu"]))
