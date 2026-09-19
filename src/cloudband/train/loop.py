"""Fastai training loop wiring for the shared protocol."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import fastai.callback.schedule  # noqa: F401  patches fine_tune onto Learner
from fastai.callback.core import Callback
from fastai.callback.tracker import SaveModelCallback
from fastai.callback.training import GradientAccumulation
from fastai.learner import Learner
from fastai.torch_core import set_seed

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

    def before_batch(self):
        gsd = sample_gsd(self.min_gsd_m, self.max_gsd_m, self.rng)
        size = patch_size_px(gsd)
        images, annotations = self.learn.xb[0], self.learn.yb[0]
        self.learn.xb = (resize_image(images, size, self.nodata_value),)
        self.learn.yb = (resize_annotation(annotations, size),)
        self.last_gsd_m = gsd
        self.last_size_px = size


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


@dataclass(frozen=True)
class FitResult:
    learner: Learner
    best_val_loss: float
    checkpoint_name: str


def checkpoint_name(protocol: TrainProtocol, seed: int) -> str:
    return f"{protocol.run_id}-seed{seed}"


def fit_protocol(
    learner: Learner,
    protocol: TrainProtocol,
    seed: int,
    nodata_value: float | None = 0.0,
) -> FitResult:
    """Run fine_tune with the hyperparameters and seed the protocol fixes.

    Gradient accumulation makes the actual optimiser step match the
    protocol's effective batch size regardless of the dataloader's own
    per-step batch size. The checkpoint tracks valid_loss, which is cross
    entropy given the loss this project trains with, matching the
    protocol's checkpoint_metric.

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
    name = checkpoint_name(protocol, seed)
    save_cb = SaveModelCallback(monitor=CHECKPOINT_MONITOR, fname=name, with_opt=False)

    learner.add_cb(resolution_cb)
    learner.add_cb(accumulation_cb)
    learner.add_cb(save_cb)
    try:
        learner.fine_tune(
            epochs=protocol.unfrozen_epochs,
            freeze_epochs=protocol.frozen_epochs,
            base_lr=protocol.learning_rate,
            wd=protocol.weight_decay,
        )
    finally:
        learner.remove_cb(save_cb)
        learner.remove_cb(accumulation_cb)
        learner.remove_cb(resolution_cb)

    return FitResult(
        learner=learner,
        best_val_loss=float(save_cb.best),
        checkpoint_name=name,
    )