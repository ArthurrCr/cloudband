import inspect

import numpy as np
import torch
from fastai.data.core import DataLoaders

from cloudband.datasets.cloudsen12 import Sample
from cloudband.train.data import build_dataloaders
from cloudband.train.lr_search import search_learning_rate
from cloudband.train.protocol import TrainProtocol

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
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=1)

    def forward(self, x):
        return self.conv(x.float())


def build_test_dls() -> DataLoaders:
    return build_dataloaders(
        train_table=range(4),
        valid_table=range(2),
        micro_batch_size=2,
        read_sample=fake_read_sample,
    )


def test_search_learning_rate_runs_once_per_grid_point():
    dls = build_test_dls()
    protocol = TrainProtocol(
        run_id="ocm-rgn-cs12-shared",
        architecture="ocm",
        learning_rate=1e-3,
        effective_batch_size=2,
    )
    grid = (1e-4, 1e-3)

    manifest = search_learning_rate(
        dls,
        protocol,
        seed=protocol.seeds[0],
        model_builder=TinyModelStandIn,
        grid=grid,
        frozen_epochs=1,
        unfrozen_epochs=1,
    )

    assert manifest.architecture == "ocm"
    assert manifest.grid == grid
    assert [run.learning_rate for run in manifest.runs] == list(grid)
    assert all(run.split == "validation" for run in manifest.runs)
    assert all(run.val_loss < float("inf") for run in manifest.runs)


def test_winner_picks_the_lowest_validation_loss_from_a_real_search():
    dls = build_test_dls()
    protocol = TrainProtocol(
        run_id="swin-rgn-cs12-shared",
        architecture="swin",
        learning_rate=1e-3,
        effective_batch_size=2,
    )

    manifest = search_learning_rate(
        dls,
        protocol,
        seed=protocol.seeds[0],
        model_builder=TinyModelStandIn,
        grid=(1e-4, 1e-3, 1e-2),
        frozen_epochs=1,
        unfrozen_epochs=1,
    )

    winner = manifest.winner()
    assert winner.val_loss == min(run.val_loss for run in manifest.runs)


def test_same_grid_and_run_count_across_architectures():
    dls = build_test_dls()
    grid = (1e-4, 1e-3)
    ocm_protocol = TrainProtocol(
        run_id="ocm-rgn-cs12-shared", architecture="ocm", learning_rate=1e-3
    )
    swin_protocol = TrainProtocol(
        run_id="swin-rgn-cs12-shared", architecture="swin", learning_rate=1e-3
    )

    ocm_manifest = search_learning_rate(
        dls,
        ocm_protocol,
        seed=0,
        model_builder=TinyModelStandIn,
        grid=grid,
        frozen_epochs=1,
        unfrozen_epochs=1,
    )
    swin_manifest = search_learning_rate(
        dls,
        swin_protocol,
        seed=0,
        model_builder=TinyModelStandIn,
        grid=grid,
        frozen_epochs=1,
        unfrozen_epochs=1,
    )

    assert len(ocm_manifest.runs) == len(swin_manifest.runs)
    ocm_rates = [run.learning_rate for run in ocm_manifest.runs]
    swin_rates = [run.learning_rate for run in swin_manifest.runs]
    assert ocm_rates == swin_rates


def test_search_learning_rate_has_no_parameter_that_could_carry_a_test_split():
    params = set(inspect.signature(search_learning_rate).parameters)
    assert "test" not in {p.lower() for p in params}
    assert "dls" in params