"""Measure what a full run costs before committing GPU days to it.

The comparison is on the order of a thousand epochs. A short run on a small
slice of the real data, through the same training path, tells how long an epoch
takes, whether a batch fits in memory, and whether the whole stack works, at
the price of a few minutes instead of a failure on day two.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from fastai.callback.core import Callback
from fastai.data.core import DataLoaders
from fastai.learner import Learner

from cloudband.train.loop import fit_protocol
from cloudband.train.loss import build_loss
from cloudband.train.protocol import (
    FULL_FROZEN_EPOCHS,
    FULL_UNFROZEN_EPOCHS,
    PROXY_FROZEN_EPOCHS,
    PROXY_UNFROZEN_EPOCHS,
    TrainProtocol,
)

SECONDS_PER_HOUR = 3600.0


def _now() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


class EpochTimer(Callback):
    """Wall-clock seconds spent in each training pass and each validation pass."""

    def __init__(self) -> None:
        super().__init__()
        self.train_seconds: list[float] = []
        self.valid_seconds: list[float] = []
        self._started = 0.0

    def before_train(self) -> None:
        self._started = _now()

    def after_train(self) -> None:
        self.train_seconds.append(_now() - self._started)

    def before_validate(self) -> None:
        self._started = _now()

    def after_validate(self) -> None:
        self.valid_seconds.append(_now() - self._started)


@dataclass(frozen=True)
class CostEstimate:
    """Seconds per full-size epoch for one model, extrapolated from a small slice."""

    label: str
    frozen_epoch_seconds: float
    unfrozen_epoch_seconds: float
    peak_memory_gb: float | None

    def hours(self, frozen_epochs: int, unfrozen_epochs: int) -> float:
        seconds = (
            frozen_epochs * self.frozen_epoch_seconds
            + unfrozen_epochs * self.unfrozen_epoch_seconds
        )
        return seconds / SECONDS_PER_HOUR

    def run_hours(self) -> float:
        """One full training run."""
        return self.hours(FULL_FROZEN_EPOCHS, FULL_UNFROZEN_EPOCHS)

    def search_hours(self, grid_size: int) -> float:
        """The learning-rate search: every candidate is a short proxy run."""
        return grid_size * self.hours(PROXY_FROZEN_EPOCHS, PROXY_UNFROZEN_EPOCHS)


def estimate_cost(
    small_dls: DataLoaders,
    full_train_samples: int,
    full_valid_samples: int,
    protocol: TrainProtocol,
    model_builder,
    label: str,
) -> CostEstimate:
    """Time a short real fit on a small slice and scale it to the full dataset.

    small_dls should hold at least protocol.effective_batch_size training
    samples, so the optimiser actually steps. The first frozen epoch absorbs
    start-up costs and is discarded; the second frozen epoch and the unfrozen
    epoch are what get scaled. The checkpoint the fit writes is deleted.
    """
    n_train = len(small_dls.train_ds)
    n_valid = len(small_dls.valid_ds)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    learner = Learner(small_dls, model_builder(), loss_func=build_loss())
    timer = EpochTimer()
    short = replace(protocol, frozen_epochs=2, unfrozen_epochs=1)
    result = fit_protocol(
        learner,
        short,
        seed=protocol.seeds[0],
        checkpoint_suffix=f"smoke-{label}",
        extra_cbs=(timer,),
    )
    checkpoint_file = f"{result.checkpoint_name}.pth"
    checkpoint = Path(learner.path) / learner.model_dir / checkpoint_file
    checkpoint.unlink(missing_ok=True)

    valid_per_sample = timer.valid_seconds[-1] / max(n_valid, 1)
    frozen_per_sample = timer.train_seconds[1] / max(n_train, 1)
    unfrozen_per_sample = timer.train_seconds[2] / max(n_train, 1)

    peak = None
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024**3

    valid_seconds = valid_per_sample * full_valid_samples
    return CostEstimate(
        label=label,
        frozen_epoch_seconds=frozen_per_sample * full_train_samples + valid_seconds,
        unfrozen_epoch_seconds=unfrozen_per_sample * full_train_samples + valid_seconds,
        peak_memory_gb=peak,
    )


def plan_hours(
    search_estimate: CostEstimate,
    run_estimates: tuple[CostEstimate, ...],
    seeds: int,
    grid_size: int,
) -> float:
    """Total hours for one architecture: one search, then every model per seed."""
    per_seed = sum(estimate.run_hours() for estimate in run_estimates)
    return search_estimate.search_hours(grid_size) + seeds * per_seed