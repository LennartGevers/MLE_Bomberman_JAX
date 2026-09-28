"""Saving and restoring agents and optimiser state with Orbax."""

from __future__ import annotations

import pathlib
from typing import Any

import equinox as eqx
import orbax.checkpoint as ocp
from jaxtyping import PyTree


class Checkpointer:
    """Saves ``(step, agents, opt_states)`` every ``every`` updates."""

    every: int = 1

    def __call__(self, step: int, agents: PyTree, opt_states: PyTree) -> None:
        raise NotImplementedError

    def wait(self) -> None:
        """Wait until all saves are written to disk."""


class NoCheckpointer(Checkpointer):
    """Checkpointer that saves nothing."""

    def __call__(self, step, agents, opt_states):
        pass


class OrbaxCheckpointer(Checkpointer):
    """Saves checkpoints in ``directory`` and keeps the ``keep`` most recent ones."""

    def __init__(
        self,
        directory: str | pathlib.Path,
        *,
        every: int = 100,
        keep: int = 3,
        save_opt_state: bool = True,
    ):
        self.every = int(every)
        self.save_opt_state = bool(save_opt_state)
        options = ocp.CheckpointManagerOptions(max_to_keep=keep, create=True)
        # Orbax needs an absolute path. A relative one would only fail later, in the
        # middle of training, so we check it here.
        self._manager = ocp.CheckpointManager(
            pathlib.Path(directory).absolute(), options=options
        )

    def __call__(self, step, agents, opt_states):
        params = eqx.filter(agents, eqx.is_array)
        args = {"agents": ocp.args.StandardSave(params)}
        if self.save_opt_state:
            args["opt_states"] = ocp.args.StandardSave(opt_states)
        self._manager.save(int(step), args=ocp.args.Composite(**args))

    def wait(self) -> None:
        self._manager.wait_until_finished()

    def latest_step(self) -> int | None:
        return self._manager.latest_step()

    def restore(
        self,
        agents_template: PyTree,
        opt_states_template: PyTree | None = None,
        *,
        step: int | None = None,
    ) -> tuple[Any, Any, int]:
        """Load a checkpoint into agents shaped like ``agents_template``.

        Returns ``(agents, opt_states, step)``. By default the latest checkpoint is loaded.
        """
        step = self.latest_step() if step is None else step
        if step is None:
            raise FileNotFoundError("no checkpoint found to restore")

        params_template = eqx.filter(agents_template, eqx.is_array)
        args = {"agents": ocp.args.StandardRestore(params_template)}
        if self.save_opt_state:
            args["opt_states"] = ocp.args.StandardRestore(opt_states_template)

        restored = self._manager.restore(step, args=ocp.args.Composite(**args))
        static = eqx.filter(agents_template, eqx.is_array, inverse=True)
        agents = eqx.combine(restored["agents"], static)
        opt_states = restored.get("opt_states", opt_states_template)
        return agents, opt_states, step

    def close(self) -> None:
        """Stop the background threads of the checkpoint manager."""
        self._manager.close()
