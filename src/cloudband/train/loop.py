"""Fastai training loop wiring for the shared protocol."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fastai.callback.schedule  # noqa: F401  patches fine_tune onto Learner
import numpy as np
import torch
import torch.nn.functional as F
from fastai.callback.core import Callback
from fastai.callback.fp16 import MixedPrecision
from fastai.callback.tracker import SaveModelCallback, TerminateOnNaNCallback
from fastai.callback.training import GradientAccumulation
from fastai.learner import Learner
from fastai.torch_core import default_device, set_seed

from cloudband.train.progress import (
    STATE_VERSION,
    EpochSaver,
    RestoreBest,
    capture_rng,
    cpu_state_dict,
    fit_phase,
    load_optimizer_state,
    optimizer_state,
    phase_specs,
    restore_rng,
)
from cloudband.train.protocol import TrainProtocol, patch_size_px
from cloudband.train.resampling import sample_gsd

BILINEAR_MODE = "bilinear"
NEAREST_MODE = "nearest"
CHECKPOINT_MONITOR = "valid_loss"


def resize_image(
    image: torch.Tensor, size: int, nodata_value: float | None = None
) -> torch.Tensor:
    """Bilinear resize on the device the tensor already lives on.

    No-data pixels are re-stamped after interpolation, for the same reason
    the numpy version does it: bilinear blends the sentinel into neighbours.
    """
    resized = F.interpolate(
        image, size=(size, size), mode=BILINEAR_MODE, align_corners=False
    )
    if nodata_value is not None:
        nodata = (image == nodata_value).float()
        resized_nodata = F.interpolate(nodata, size=(size, size), mode=NEAREST_MODE) > 0
        resized = torch.where(
            resized_nodata, torch.full_like(resized, nodata_value), resized
        )
    return resized


def resize_annotation(annotation: torch.Tensor, size: int) -> torch.Tensor:
    """Nearest-neighbour resize, so no intermediate class value is invented."""
    with_channel = annotation.unsqueeze(1).float()
    resized = F.interpolate(with_channel, size=(size, size), mode=NEAREST_MODE)
    return resized.squeeze(1).to(annotation.dtype)


class MixedResolutionCallback(Callback):
    """Resamples every batch to one GSD drawn at random from a range."""

    def __init__(
        self,
        min_gsd_m: float,
        max_gsd_m: float,
        rng: np.random.Generator,
        nodata_value: float | None = None,
    ):
        self.min_gsd_m = min_gsd_m
        self.max_gsd_m = max_gsd_m
        self.rng = rng
        self.nodata_value = nodata_value
        self.last_gsd_m: float | None = None
        self.last_size_px: int | None = None
        self.gsd_history: list[float] = []

    def before_batch(self):
        gsd = sample_gsd(self.min_gsd_m, self.max_gsd_m, self.rng)
        size = patch_size_px(gsd)
        images, annotations = self.learn.xb[0], self.learn.yb[0]
        self.learn.xb = (resize_image(images, size, self.nodata_value),)
        self.learn.yb = (resize_annotation(annotations, size),)
        self.last_gsd_m = gsd
        self.last_size_px = size
        self.gsd_history.append(gsd)


def mixed_resolution_callback(
    protocol: TrainProtocol,
    seed: int,
    nodata_value: float | None = 0.0,
) -> MixedResolutionCallback:
    return MixedResolutionCallback(
        min_gsd_m=protocol.min_gsd_m,
        max_gsd_m=protocol.max_gsd_m,
        rng=np.random.default_rng(seed),
        nodata_value=nodata_value,
    )


class ValidLossHistory(Callback):
    """Validation loss after every epoch, frozen and unfrozen phases together.

    Recorder.values restarts at each fit and fine_tune is two fits, so it only
    holds the unfrozen phase by the time training ends.
    """

    order = 70  # after the Recorder (50) and the trackers (60)

    def __init__(self) -> None:
        super().__init__()
        self.values: list[float] = []

    def after_epoch(self) -> None:
        self.values.append(float(self.learn.recorder.values[-1][1]))


@dataclass(frozen=True)
class FitResult:
    learner: Learner
    best_val_loss: float
    checkpoint_name: str
    seed: int
    sampled_gsds: tuple
    valid_losses: tuple = ()


def _cuda_available() -> bool:
    return torch.cuda.is_available()


def _use_mixed_precision(protocol: TrainProtocol) -> bool:
    """Mixed precision needs a GPU; without one the flag is ignored."""
    return protocol.mixed_precision and _cuda_available()


def checkpoint_name(protocol: TrainProtocol, seed: int, suffix: str = "") -> str:
    base = f"{protocol.run_id}-seed{seed}"
    return f"{base}-{suffix}" if suffix else base


class TrainingDiverged(RuntimeError):
    """The loss became NaN or infinite, so the run did not finish its epochs."""


def _check_resume(state: dict, fingerprint: dict) -> None:
    """Refuse to continue from progress that was made under other settings."""
    if state.get("version") != STATE_VERSION:
        raise RuntimeError(
            f"{fingerprint['name']}: saved progress has format version "
            f"{state.get('version')}, this code reads {STATE_VERSION}"
        )
    if state["fingerprint"] != fingerprint:
        differing = sorted(
            key
            for key in fingerprint
            if state["fingerprint"].get(key) != fingerprint[key]
        )
        raise RuntimeError(
            f"{fingerprint['name']}: saved progress was made with other settings "
            f"({', '.join(differing)}); delete it to start the run again, or "
            "restore the settings it was made with"
        )


def fit_protocol(
    learner: Learner,
    protocol: TrainProtocol,
    seed: int,
    nodata_value: float | None = 0.0,
    checkpoint_suffix: str = "",
    extra_cbs: tuple = (),
    progress_store=None,
) -> FitResult:
    """Run fine_tune with the hyperparameters and seed the protocol fixes.

    fine_tune is run phase by phase here (see progress.phase_specs), which gives the
    same training, and that is what lets a run stop after any epoch. With a
    progress_store, the state is saved after every epoch and a run that finds one
    continues from it. The store needs load_epoch_state(name), save_epoch_state(name,
    state) and, optionally, heartbeat(name); RunStore has them. A saved state that
    belongs to other settings is refused rather than silently continued.

    Gradient accumulation makes the actual optimiser step match the
    protocol's effective batch size regardless of the dataloader's own
    per-step batch size. The checkpoint tracks valid_loss, which is cross
    entropy given the loss this project trains with, matching the
    protocol's checkpoint_metric.

    checkpoint_suffix distinguishes multiple models trained under the same
    run_id and seed, such as the separate backbones in an ensemble; without
    it, two such calls would write to the same checkpoint file and the
    second would silently overwrite the first.

    Mixed precision is applied when the protocol asks for it and a GPU exists. A
    loss that turns NaN or infinite stops the fit, and a fit that ran fewer
    epochs than the protocol raises instead of returning a result.

    extra_cbs are attached for the duration of the fit and removed after, for
    instrumentation such as timing; they must not change what is trained.

    nodata_value defaults to 0.0, not the raw sentinel: by the time a batch
    reaches this function it has already gone through dynamic_z_score,
    which zeroes no-data pixels rather than leaving the raw sentinel in
    place. This is an approximation, not an exact marker — a real pixel
    that happens to normalize to exactly 0.0 would also get treated as
    no-data during the resize. No-data regions in this data are large
    contiguous blocks, not scattered pixels, so the practical impact of
    that collision is expected to be small.
    """
    if seed not in protocol.seeds:
        raise ValueError(
            f"seed {seed} is not one of the protocol's seeds {protocol.seeds}"
        )

    set_seed(seed, reproducible=True)
    resolution_cb = mixed_resolution_callback(
        protocol, seed=seed, nodata_value=nodata_value
    )
    accumulation_cb = GradientAccumulation(n_acc=protocol.effective_batch_size)
    name = checkpoint_name(protocol, seed, checkpoint_suffix)
    save_cb = SaveModelCallback(monitor=CHECKPOINT_MONITOR, fname=name, with_opt=False)
    history_cb = ValidLossHistory()
    guard_cbs = [TerminateOnNaNCallback()]
    if _use_mixed_precision(protocol):
        guard_cbs.append(MixedPrecision())

    fingerprint = {
        "name": name,
        "seed": seed,
        "learning_rate": protocol.learning_rate,
        "frozen_epochs": protocol.frozen_epochs,
        "unfrozen_epochs": protocol.unfrozen_epochs,
        "weight_decay": protocol.weight_decay,
        "effective_batch_size": protocol.effective_batch_size,
        "min_gsd_m": protocol.min_gsd_m,
        "max_gsd_m": protocol.max_gsd_m,
    }
    resume = progress_store.load_epoch_state(name) if progress_store else None
    if resume is not None:
        _check_resume(resume, fingerprint)

    best_path = Path(learner.path) / learner.model_dir / f"{name}.pth"

    def snapshot(phase: str, epochs_done: int) -> dict:
        """Everything needed to continue from the end of this epoch."""
        started = epochs_done > 0
        weights = None
        if started and best_path.is_file():
            weights = torch.load(best_path, map_location="cpu", weights_only=False)
        return {
            "version": STATE_VERSION,
            "fingerprint": fingerprint,
            "phase": phase,
            "epochs_done": epochs_done,
            "model": cpu_state_dict(learner.model),
            "opt": optimizer_state(learner.opt) if started else None,
            "best": float(save_cb.best) if started else float("inf"),
            "best_weights": weights,
            "valid_losses": list(history_cb.values),
            "gsd_history": list(resolution_cb.gsd_history),
            "rng": capture_rng(resolution_cb),
        }

    def save(state: dict) -> None:
        progress_store.save_epoch_state(name, state)
        heartbeat = getattr(progress_store, "heartbeat", None)
        if heartbeat is not None:
            heartbeat(name)

    learner.add_cb(resolution_cb)
    learner.add_cb(accumulation_cb)
    learner.add_cb(save_cb)
    learner.add_cb(history_cb)
    for callback in guard_cbs:
        learner.add_cb(callback)
    for callback in extra_cbs:
        learner.add_cb(callback)
    try:
        specs = phase_specs(protocol)
        first_phase = 0
        done = 0
        if resume is not None:
            # fastai only moves the model to the GPU when a fit starts. The
            # optimiser state is loaded before that and is placed where the
            # parameters are at that moment, so the model goes to the device first;
            # otherwise the momentum buffers stay on the CPU and the first step
            # fails with "found at least two devices".
            learner.model.to(getattr(learner.dls, "device", default_device()))
            learner.model.load_state_dict(resume["model"])
            history_cb.values = list(resume["valid_losses"])
            resolution_cb.gsd_history = list(resume["gsd_history"])
            first_phase = [spec.name for spec in specs].index(resume["phase"])
            done = resume["epochs_done"]

        for index, spec in enumerate(specs):
            if index < first_phase:
                continue
            already = done if index == first_phase else 0
            (learner.freeze if spec.name == "frozen" else learner.unfreeze)()
            callbacks = []
            if resume is not None and index == first_phase and already > 0:
                if resume["opt"] is not None:
                    load_optimizer_state(learner.opt, resume["opt"])
                callbacks.append(
                    RestoreBest(
                        save_cb, resume["best"], resume["best_weights"], best_path
                    )
                )
            if resume is not None and index == first_phase:
                restore_rng(resume["rng"], resolution_cb)
            if progress_store is not None:
                callbacks.append(EpochSaver(snapshot, save, spec.name, already))
            if already >= spec.epochs:
                # stopped after the last epoch of this phase: do what the end of a
                # fit does, which is to go back to the best weights of the phase
                if resume["best_weights"] is not None:
                    learner.model.load_state_dict(resume["best_weights"])
                save_cb.best = resume["best"]
            else:
                fit_phase(learner, spec, already, protocol.weight_decay, callbacks)
            if progress_store is not None and index + 1 < len(specs):
                save(snapshot(specs[index + 1].name, 0))
    except FileNotFoundError as error:
        # With a NaN loss from the first epoch no checkpoint is ever written, and
        # fastai fails reloading the best one; say what actually happened.
        if Path(str(error.filename)).name != f"{name}.pth":
            raise
        raise TrainingDiverged(
            f"{name}: no checkpoint was written because the loss was NaN or "
            "infinite from the first epoch"
        ) from error
    finally:
        for callback in extra_cbs:
            learner.remove_cb(callback)
        for callback in guard_cbs:
            learner.remove_cb(callback)
        learner.remove_cb(history_cb)
        learner.remove_cb(save_cb)
        learner.remove_cb(accumulation_cb)
        learner.remove_cb(resolution_cb)

    expected_epochs = protocol.frozen_epochs + protocol.unfrozen_epochs
    if len(history_cb.values) != expected_epochs:
        raise TrainingDiverged(
            f"{name}: training ended after {len(history_cb.values)} of "
            f"{expected_epochs} epochs; a loss that is NaN or infinite stops it, "
            "so this run must not be treated as finished"
        )

    return FitResult(
        learner=learner,
        best_val_loss=float(save_cb.best),
        checkpoint_name=name,
        seed=seed,
        sampled_gsds=tuple(resolution_cb.gsd_history),
        valid_losses=tuple(history_cb.values),
    )