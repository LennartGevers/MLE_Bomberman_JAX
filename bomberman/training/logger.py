"""Loggers for training progress. They are called from the compiled training loop."""

from __future__ import annotations

from collections.abc import Mapping


class Logger:
    """Receives ``(iteration, metrics)`` every ``every`` updates."""

    every: int = 1

    def __call__(self, iteration: int, metrics: Mapping[str, float]) -> None:
        raise NotImplementedError


class NoLogger(Logger):
    """Logger that logs nothing."""

    def __call__(self, iteration, metrics):
        pass


class PrintLogger(Logger):
    """Prints one line per logged update, optionally only the metrics in ``keys``."""

    def __init__(self, every: int = 10, keys: tuple[str, ...] | None = None):
        self.every = int(every)
        self.keys = keys

    def __call__(self, iteration, metrics):
        keys = self.keys if self.keys is not None else sorted(metrics)
        body = "  ".join(f"{k}={float(metrics[k]):+.3f}" for k in keys if k in metrics)
        print(f"[{int(iteration):5d}] {body}")


class HistoryLogger(Logger):
    """Stores every logged update in ``rows``, e.g. for plots."""

    def __init__(self, every: int = 1):
        self.every = int(every)
        self.rows: list[dict[str, float]] = []

    def __call__(self, iteration, metrics):
        self.rows.append({"iteration": int(iteration), **{k: float(v) for k, v in metrics.items()}})


class WandbLogger(Logger):
    """Logs each update to an existing Weights & Biases run."""

    def __init__(self, run, every: int = 1):
        self.run = run
        self.every = int(every)

    def __call__(self, iteration, metrics):
        self.run.log({k: float(v) for k, v in metrics.items()}, step=int(iteration))


class MultiLogger(Logger):
    """Passes updates on to several loggers."""

    def __init__(self, *loggers: Logger):
        self.loggers = loggers
        self.every = min((logger.every for logger in loggers), default=1)

    def __call__(self, iteration, metrics):
        for logger in self.loggers:
            if int(iteration) % logger.every == 0:
                logger(iteration, metrics)
