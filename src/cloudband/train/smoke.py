"""Measure what a full run costs before committing GPU days to it.

The comparison is on the order of a thousand epochs. A short run on a small
slice of the real data, through the same training path, tells how long an epoch
takes, whether a batch fits in memory, and whether the whole stack works, at
the price of a few minutes instead of a failure on day two.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.nn.functional as F
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
    patch_size_px,
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

    # The next model must not start with this one still on the GPU.
    del learner, result
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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


MEMORY_ERROR_MARKERS = (
    "out of memory",
    "device_allocation_failed",
    "cudnn_status_alloc",
)


def _is_memory_error(error: BaseException) -> bool:
    """CUDA reports a full GPU either as OutOfMemoryError or, inside cuDNN, as a
    RuntimeError whose text names the failed allocation."""
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(error, RuntimeError) and any(
        marker in str(error).lower() for marker in MEMORY_ERROR_MARKERS
    )


def _fits_in_memory(
    model_builder,
    batch_size: int,
    size: int,
    mixed_precision: bool,
    headroom: float,
    device: str = "cuda",
) -> bool:
    """One full training step on random data at the given size, on the device.

    Everything trainable is unfrozen, as in the second phase of a run, and the
    optimiser takes a step so that its state is in memory too. True when the step
    runs and its peak memory stays under headroom times the device's memory.
    """
    on_gpu = device == "cuda"
    model = optimizer = x = y = loss = None
    try:
        if on_gpu:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        model = model_builder().to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        x = torch.rand(batch_size, 3, size, size, device=device)
        y = torch.randint(0, 4, (batch_size, size, size), device=device)
        amp = mixed_precision and on_gpu
        with torch.autocast(device, dtype=torch.float16, enabled=amp):
            loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        if not on_gpu:
            return True
        torch.cuda.synchronize()
        total = torch.cuda.get_device_properties(0).total_memory
        return torch.cuda.max_memory_allocated() <= headroom * total
    except (RuntimeError, torch.cuda.OutOfMemoryError) as error:
        if _is_memory_error(error):
            return False
        raise
    finally:
        del model, optimizer, x, y, loss
        gc.collect()
        if on_gpu:
            torch.cuda.empty_cache()


def probe_micro_batch_size(
    model_builder,
    protocol: TrainProtocol,
    candidates: tuple[int, ...] = (8, 4, 2),
    headroom: float = 0.85,
    fits=None,
) -> int:
    """The largest micro-batch whose worst case still fits in GPU memory.

    Training draws a resolution for every batch, and the smallest ground sampling
    distance gives the largest tensor (565 px at 9 m). A probe that merely trains a
    few batches passes or fails by the luck of the draw, so this one forces that
    size. Memory is the only thing it decides: the protocol's effective batch size
    is reached by gradient accumulation whatever the micro-batch is.

    Without a GPU there is nothing to measure and the first candidate is returned.
    `fits` replaces the measurement, for tests.
    """
    if fits is None and not torch.cuda.is_available():
        return candidates[0]
    size = patch_size_px(protocol.min_gsd_m)

    def measured(batch_size: int) -> bool:
        return _fits_in_memory(
            model_builder, batch_size, size, protocol.mixed_precision, headroom
        )

    check = fits or measured
    for batch_size in candidates:
        if check(batch_size):
            return batch_size
    raise RuntimeError(
        f"even a micro-batch of {candidates[-1]} does not fit in GPU memory at "
        f"{size} px for {protocol.run_id}"
    )


def loader_samples_per_second(dls, max_batches: int = 16) -> float:
    """How fast the input pipeline alone delivers training samples, GPU idle.

    The first batch is taken before timing starts, so worker start-up is not part
    of the rate. Compare it with how fast the model consumes samples: whichever is
    lower sets the speed of an epoch.
    """
    iterator = iter(dls.train)
    next(iterator)
    seen = 0
    started = time.perf_counter()
    for _, batch in zip(range(max_batches), iterator):
        seen += len(batch[0])
    elapsed = time.perf_counter() - started
    return seen / elapsed if elapsed > 0 and seen else 0.0