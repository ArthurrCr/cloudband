import numpy as np
import torch

from cloudband.datasets.cloudsen12 import Sample
from cloudband.models import ocm as ocm_module
from cloudband.models import swin_upernet as swin_module
from cloudband.models.ocm import PAPER_BACKBONES
from cloudband.train.data import build_dataloaders
from cloudband.train.protocol import LR_SEARCH_GRID, TrainProtocol

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


def build_test_dls():
    return build_dataloaders(
        train_table=range(4),
        valid_table=range(2),
        micro_batch_size=2,
        read_sample=fake_read_sample,
    )


def test_ocm_ensemble_phase2_searches_once_and_trains_both_backbones(monkeypatch):
    monkeypatch.setattr(ocm_module, "build_unet", lambda *a, **k: TinyModelStandIn())
    dls = build_test_dls()
    protocol = TrainProtocol(
        run_id="ocm-rgn-cs12-shared",
        architecture="ocm",
        learning_rate=1e-3,
        effective_batch_size=2,
    )

    runs = ocm_module.run_ocm_ensemble_phase2(dls, protocol, seed=protocol.seeds[0])

    assert set(runs) == set(PAPER_BACKBONES)
    winning_rates = {run.winning_protocol.learning_rate for run in runs.values()}
    assert len(winning_rates) == 1, "both backbones must train at the same winning rate"
    for run in runs.values():
        assert len(run.lr_search.runs) == len(LR_SEARCH_GRID)
        assert run.fit_result.best_val_loss < float("inf")


def test_swin_phase2_searches_and_trains(monkeypatch):
    monkeypatch.setattr(
        swin_module, "build_swin_upernet", lambda *a, **k: TinyModelStandIn()
    )
    dls = build_test_dls()
    protocol = TrainProtocol(
        run_id="swin-rgn-cs12-shared",
        architecture="swin",
        learning_rate=1e-3,
        effective_batch_size=2,
    )

    run = swin_module.run_swin_upernet_phase2(dls, protocol, seed=protocol.seeds[0])

    assert len(run.lr_search.runs) == len(LR_SEARCH_GRID)
    assert run.fit_result.best_val_loss < float("inf")
    assert run.manifest.seed == protocol.seeds[0]