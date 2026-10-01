from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cloudband.datasets.cloudsen12 import Sample
from cloudband.train import smoke
from cloudband.train.data import build_dataloaders
from cloudband.train.protocol import (
    FULL_FROZEN_EPOCHS,
    FULL_UNFROZEN_EPOCHS,
    PROXY_FROZEN_EPOCHS,
    PROXY_UNFROZEN_EPOCHS,
    TrainProtocol,
)
from cloudband.train.smoke import CostEstimate, estimate_cost, plan_hours

PATCH_SIZE = 32


def fake_read_sample(table, index, crop_to_valid=True):
    rng = np.random.default_rng(index)
    image = rng.integers(0, 10000, size=(13, PATCH_SIZE, PATCH_SIZE))
    annotation = rng.integers(0, 4, size=(PATCH_SIZE, PATCH_SIZE))
    return Sample(
        identifier=f"scene-{index}",
        image=image.astype(np.int32),
        annotation=annotation.astype(np.int32),
    )


class TinyModelStandIn(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=1)

    def forward(self, x):
        return self.conv(x.float())


def test_hours_follow_the_epoch_budget():
    estimate = CostEstimate("m", frozen_epoch_seconds=10, unfrozen_epoch_seconds=20,
                            peak_memory_gb=None)

    full = FULL_FROZEN_EPOCHS * 10 + FULL_UNFROZEN_EPOCHS * 20
    proxy = PROXY_FROZEN_EPOCHS * 10 + PROXY_UNFROZEN_EPOCHS * 20
    assert estimate.run_hours() == pytest.approx(full / 3600)
    # every candidate of the grid is one proxy run
    assert estimate.search_hours(grid_size=5) == pytest.approx(5 * proxy / 3600)


def test_plan_hours_is_one_search_plus_every_model_per_seed():
    estimate = CostEstimate("m", 10, 20, None)
    other = CostEstimate("n", 30, 60, None)

    total = plan_hours(estimate, (estimate, other), seeds=5, grid_size=5)

    expected = estimate.search_hours(5) + 5 * (estimate.run_hours() + other.run_hours())
    assert total == pytest.approx(expected)


def test_estimate_cost_returns_positive_times_and_leaves_no_checkpoint(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    dls = build_dataloaders(
        train_table=range(8),
        valid_table=range(4),
        micro_batch_size=2,
        read_sample=fake_read_sample,
    )
    protocol = TrainProtocol(
        run_id="swin-rgn-cs12-shared",
        architecture="swin",
        learning_rate=1e-3,
        effective_batch_size=4,
    )

    estimate = estimate_cost(
        dls,
        full_train_samples=16980,
        full_valid_samples=1070,
        protocol=protocol,
        model_builder=TinyModelStandIn,
        label="tiny",
    )

    assert estimate.frozen_epoch_seconds > 0
    assert estimate.unfrozen_epoch_seconds > 0
    if torch.cuda.is_available():
        assert estimate.peak_memory_gb is not None and estimate.peak_memory_gb > 0
    else:
        assert estimate.peak_memory_gb is None
    assert not list((tmp_path / "models").glob("*smoke*"))   # the probe cleans up


def test_estimate_scales_with_the_full_dataset_size(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dls = build_dataloaders(
        train_table=range(8),
        valid_table=range(4),
        micro_batch_size=2,
        read_sample=fake_read_sample,
    )
    protocol = TrainProtocol(
        run_id="swin-rgn-cs12-shared",
        architecture="swin",
        learning_rate=1e-3,
        effective_batch_size=4,
    )
    kwargs = dict(protocol=protocol, model_builder=TinyModelStandIn, label="tiny")

    small = estimate_cost(dls, 1000, 100, **kwargs)
    large = estimate_cost(dls, 100000, 10000, **kwargs)

    # timings are noisy, but a 100x larger dataset cannot look cheaper
    assert large.frozen_epoch_seconds > small.frozen_epoch_seconds


def test_peak_memory_is_reported_in_gib_when_cuda_is_available(tmp_path, monkeypatch):
    # Exercise the GPU branch without a GPU: only the four torch.cuda calls the
    # module makes are replaced, so fastai keeps running on the CPU. The module is
    # imported at the top of this file, like estimate_cost, so both are always the
    # same object even if another test clears sys.modules (reload_package does).
    fake_cuda = SimpleNamespace(
        is_available=lambda: True,
        synchronize=lambda: None,
        reset_peak_memory_stats=lambda: None,
        max_memory_allocated=lambda: 3 * 1024**3,
    )
    monkeypatch.setattr(smoke, "torch", SimpleNamespace(cuda=fake_cuda))
    monkeypatch.chdir(tmp_path)
    dls = build_dataloaders(
        train_table=range(8),
        valid_table=range(4),
        micro_batch_size=2,
        read_sample=fake_read_sample,
    )
    protocol = TrainProtocol(
        run_id="swin-rgn-cs12-shared",
        architecture="swin",
        learning_rate=1e-3,
        effective_batch_size=4,
    )

    estimate = estimate_cost(
        dls, 1000, 100, protocol, TinyModelStandIn, label="tiny"
    )

    assert estimate.peak_memory_gb == pytest.approx(3.0)