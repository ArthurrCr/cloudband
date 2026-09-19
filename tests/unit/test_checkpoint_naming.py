import numpy as np
import torch

from cloudband.models import ocm as ocm_module
from cloudband.train.data import build_dataloaders
from cloudband.train.protocol import TrainProtocol
from cloudband.datasets.cloudsen12 import Sample

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


def test_ocm_ensemble_backbones_and_seeds_get_distinct_checkpoint_names(monkeypatch):
    monkeypatch.setattr(ocm_module, "build_unet", lambda *a, **k: TinyModelStandIn())
    dls = build_dataloaders(
        train_table=range(4),
        valid_table=range(2),
        micro_batch_size=2,
        read_sample=fake_read_sample,
    )
    protocol = TrainProtocol(
        run_id="ocm-rgn-cs12-shared",
        architecture="ocm",
        learning_rate=1e-3,
        effective_batch_size=2,
    )

    runs = ocm_module.run_ocm_ensemble_phase2(dls, protocol, seeds=(0, 1))

    names = [
        run.fit_result.checkpoint_name
        for backbone_runs in runs.values()
        for run in backbone_runs.values()
    ]
    assert len(names) == len(set(names)), f"checkpoint names collided: {names}"