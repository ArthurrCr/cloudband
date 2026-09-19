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


def test_ocm_ensemble_phase2_searches_once_and_trains_both_backbones_per_seed(
    monkeypatch,
):
    monkeypatch.setattr(ocm_module, "build_unet", lambda *a, **k: TinyModelStandIn())
    dls = build_test_dls()
    protocol = TrainProtocol(
        run_id="ocm-rgn-cs12-shared",
        architecture="ocm",
        learning_rate=1e-3,
        effective_batch_size=2,
    )
    seeds = (0, 1)

    runs = ocm_module.run_ocm_ensemble_phase2(dls, protocol, seeds=seeds)

    assert set(runs) == set(PAPER_BACKBONES)
    for backbone_runs in runs.values():
        assert set(backbone_runs) == set(seeds)

    all_runs = [
        run for backbone_runs in runs.values() for run in backbone_runs.values()
    ]
    winning_rates = {run.winning_protocol.learning_rate for run in all_runs}
    assert len(winning_rates) == 1, (
        "every backbone and seed must share the winning rate"
    )
    assert len({id(run.lr_search) for run in all_runs}) == 1, "the search must run once"

    for run in all_runs:
        assert len(run.lr_search.runs) == len(LR_SEARCH_GRID)
        assert run.fit_result.best_val_loss < float("inf")


def test_swin_phase2_searches_once_and_trains_once_per_seed(monkeypatch):
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
    seeds = (0, 1)

    results = swin_module.run_swin_upernet_phase2(dls, protocol, seeds=seeds)

    assert [r.fit_result.seed for r in results] == list(seeds)
    assert len({id(r.lr_search) for r in results}) == 1, "the search must run once"
    winning_rates = {r.winning_protocol.learning_rate for r in results}
    assert len(winning_rates) == 1

    for result in results:
        assert len(result.lr_search.runs) == len(LR_SEARCH_GRID)
        assert result.fit_result.best_val_loss < float("inf")
        assert result.manifest.seed == result.fit_result.seed