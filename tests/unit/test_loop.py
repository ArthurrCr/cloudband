import fastai.callback.schedule  # noqa: F401  patches fine_tune onto Learner
import numpy as np
import pytest
import torch
from fastai.callback.core import Callback
from fastai.callback.training import GradientAccumulation
from fastai.data.core import DataLoaders
from fastai.learner import Learner
from fastai.losses import CrossEntropyLossFlat

from cloudband.train.loop import (
    MixedResolutionCallback,
    fit_protocol,
    resize_annotation,
    resize_image,
)
from cloudband.train.protocol import TrainProtocol, ocm_shared_protocol

BATCH, CHANNELS, NATIVE_SIZE, CLASSES = 4, 3, 32, 4


class OneByOneConv(torch.nn.Module):
    """A model whose output spatial size always matches its input."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(CHANNELS, CLASSES, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def toy_dataloaders():
    images = torch.rand(BATCH * 2, CHANNELS, NATIVE_SIZE, NATIVE_SIZE)
    annotations = torch.randint(0, CLASSES, (BATCH * 2, NATIVE_SIZE, NATIVE_SIZE))
    dataset = torch.utils.data.TensorDataset(images, annotations)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH)
    return DataLoaders(loader, loader)


def toy_learner(cbs=None):
    return Learner(
        toy_dataloaders(),
        OneByOneConv(),
        loss_func=CrossEntropyLossFlat(axis=1),
        cbs=cbs or [],
    )


def test_resize_image_matches_target_size():
    image = torch.rand(BATCH, CHANNELS, NATIVE_SIZE, NATIVE_SIZE)
    resized = resize_image(image, size=17)
    assert resized.shape == (BATCH, CHANNELS, 17, 17)


def test_resize_annotation_matches_target_size_and_stays_integer():
    annotation = torch.randint(0, CLASSES, (BATCH, NATIVE_SIZE, NATIVE_SIZE))
    resized = resize_annotation(annotation, size=17)
    assert resized.shape == (BATCH, 17, 17)
    assert resized.dtype == annotation.dtype


def test_resize_annotation_never_invents_class_values():
    annotation = torch.randint(0, CLASSES, (BATCH, NATIVE_SIZE, NATIVE_SIZE))
    resized = resize_annotation(annotation, size=17)
    assert set(resized.unique().tolist()) <= set(annotation.unique().tolist())


def test_resize_image_restamps_nodata_after_bilinear_blending():
    image = torch.full((1, 1, NATIVE_SIZE, NATIVE_SIZE), 50.0)
    image[:, :, :8, :8] = 99.0
    resized = resize_image(image, size=17, nodata_value=99.0)
    assert (resized == 99.0).any()
    valid = resized[resized != 99.0]
    assert torch.allclose(valid, torch.full_like(valid, 50.0), atol=1e-3)


def test_mixed_resolution_callback_resizes_xb_and_yb_together():
    rng = np.random.default_rng(0)
    cb = MixedResolutionCallback(min_gsd_m=9.0, max_gsd_m=22.0, rng=rng)
    learn = toy_learner(cbs=[cb])
    learn.fit(1)
    assert cb.last_size_px is not None
    assert 231 <= cb.last_size_px <= 565


def test_fit_protocol_rejects_a_seed_outside_the_protocol():
    learn = toy_learner()
    protocol = ocm_shared_protocol(learning_rate=1e-3)
    with pytest.raises(ValueError):
        fit_protocol(learn, protocol, seed=999)


def test_fit_protocol_runs_and_removes_the_callbacks_afterward():
    learn = toy_learner()
    protocol = TrainProtocol(
        run_id="ocm-rgn-cs12-shared",
        architecture="ocm",
        learning_rate=1e-3,
        frozen_epochs=1,
        unfrozen_epochs=1,
        effective_batch_size=16,
    )

    fit_protocol(learn, protocol, seed=protocol.seeds[0])

    assert not any(isinstance(cb, MixedResolutionCallback) for cb in learn.cbs)
    assert not any(isinstance(cb, GradientAccumulation) for cb in learn.cbs)


def test_fit_protocol_sets_gradient_accumulation_to_the_effective_batch_size():
    seen_n_acc = []

    class RecordingAccumulation(Callback):
        def before_fit(self):
            for cb in self.learn.cbs:
                if isinstance(cb, GradientAccumulation):
                    seen_n_acc.append(cb.n_acc)

    learn = toy_learner(cbs=[RecordingAccumulation()])
    protocol = TrainProtocol(
        run_id="ocm-rgn-cs12-shared",
        architecture="ocm",
        learning_rate=1e-3,
        frozen_epochs=1,
        unfrozen_epochs=1,
        effective_batch_size=16,
    )

    fit_protocol(learn, protocol, seed=protocol.seeds[0])

    assert seen_n_acc
    assert all(value == 16 for value in seen_n_acc)