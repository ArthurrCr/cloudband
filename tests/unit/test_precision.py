from dataclasses import replace

import pytest
import torch
from fastai.callback.core import Callback
from fastai.data.core import DataLoaders
from fastai.losses import CrossEntropyLossFlat
from fastai.learner import Learner

from cloudband.train import loop
from cloudband.train.loop import fit_protocol
from cloudband.train.manifest import build_training_manifest
from cloudband.train.protocol import ocm_shared_protocol, swin_shared_protocol


class OneByOneConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class NanModel(OneByOneConv):
    def forward(self, x):
        return self.conv(x) * float("nan")


class NanAfterAWhile(OneByOneConv):
    """Fine for the first epoch, then the loss turns NaN."""

    def __init__(self, good_calls):
        super().__init__()
        self.good_calls = good_calls
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        out = self.conv(x)
        return out if self.calls <= self.good_calls else out * float("nan")


def toy_learner(model=None):
    images = torch.rand(8, 3, 16, 16)
    annotations = torch.randint(0, 4, (8, 16, 16))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(images, annotations), batch_size=4
    )
    dls = DataLoaders(loader, loader)
    return Learner(dls, model or OneByOneConv(), loss_func=CrossEntropyLossFlat(axis=1))


def small(protocol):
    return replace(protocol, frozen_epochs=1, unfrozen_epochs=1, effective_batch_size=4)


class FakePrecision(Callback):
    """Stands in for fastai's MixedPrecision so the wiring can be seen on a CPU."""

    seen_during_fit: list = []

    def before_fit(self):
        FakePrecision.seen_during_fit.append(True)


def test_mixed_precision_is_on_by_default_and_identical_for_both_architectures():
    ocm, swin = ocm_shared_protocol(1e-3), swin_shared_protocol(1e-3)

    assert ocm.mixed_precision is True
    assert ocm.mixed_precision == swin.mixed_precision


def test_the_callback_is_attached_when_asked_for_and_a_gpu_exists(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(loop, "_cuda_available", lambda: True)
    monkeypatch.setattr(loop, "MixedPrecision", FakePrecision)
    FakePrecision.seen_during_fit = []
    learner = toy_learner()

    fit_protocol(learner, small(ocm_shared_protocol(1e-3)), seed=0)

    assert FakePrecision.seen_during_fit
    assert not any(isinstance(cb, FakePrecision) for cb in learner.cbs)


def test_the_callback_is_not_attached_when_the_protocol_turns_it_off(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(loop, "_cuda_available", lambda: True)
    monkeypatch.setattr(loop, "MixedPrecision", FakePrecision)
    FakePrecision.seen_during_fit = []
    protocol = replace(small(ocm_shared_protocol(1e-3)), mixed_precision=False)

    fit_protocol(toy_learner(), protocol, seed=0)

    assert FakePrecision.seen_during_fit == []


def test_without_a_gpu_the_flag_is_ignored(monkeypatch):
    monkeypatch.setattr(loop, "_cuda_available", lambda: False)

    assert loop._use_mixed_precision(ocm_shared_protocol(1e-3)) is False


def test_a_nan_loss_raises_instead_of_returning_a_finished_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    protocol = small(ocm_shared_protocol(1e-3))

    with pytest.raises(RuntimeError, match="NaN or infinite"):
        fit_protocol(toy_learner(NanModel()), protocol, seed=0)


def test_the_precision_setting_is_recorded_in_the_manifest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    protocol = small(ocm_shared_protocol(1e-3))
    result = fit_protocol(toy_learner(), protocol, seed=0)

    manifest = build_training_manifest(result, protocol)

    assert manifest.config["mixed_precision"] is True


def test_a_loss_that_turns_nan_after_some_epochs_is_not_a_finished_run(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    protocol = small(ocm_shared_protocol(1e-3))
    # 2 training and 2 validation batches per epoch: epoch 1 is clean, epoch 2 is NaN
    model = NanAfterAWhile(good_calls=4)

    with pytest.raises(RuntimeError, match="ended after 1 of 2 epochs"):
        fit_protocol(toy_learner(model), protocol, seed=0)