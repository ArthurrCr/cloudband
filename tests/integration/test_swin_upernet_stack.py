import numpy as np
import torch

from cloudband.datasets.cloudsen12 import Sample
from cloudband.models.swin_upernet import train_swin_upernet
from cloudband.train.data import build_dataloaders
from cloudband.train.protocol import TrainProtocol

PATCH_SIZE = 64


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
    """A cheap stand-in for the real Swin+UPerNet model.

    The real model's geometry and gradient flow are verified separately, at
    the sizes the mixed-resolution protocol actually uses, in
    test_swin_upernet_model.py; this stands in only so the orchestration
    test (fit_protocol wired to a real Learner and DataLoaders) stays light
    enough for the sandbox.
    """

    def __init__(self, img_size, pretrained):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=1)

    def forward(self, x):
        return self.conv(x.float())


def fake_model_builder(img_size, pretrained):
    return TinyModelStandIn(img_size, pretrained)


def test_train_swin_upernet_end_to_end():
    dls = build_dataloaders(
        train_table=range(4),
        valid_table=range(2),
        micro_batch_size=2,
        read_sample=fake_read_sample,
    )
    protocol = TrainProtocol(
        run_id="swin-rgn-cs12-shared",
        architecture="swin",
        learning_rate=1e-4,
        frozen_epochs=1,
        unfrozen_epochs=1,
        effective_batch_size=2,
    )

    result = train_swin_upernet(
        dls,
        protocol,
        seed=protocol.seeds[0],
        img_size=PATCH_SIZE,
        pretrained=False,
        model_builder=fake_model_builder,
    )

    assert result.learner.recorder.losses
    assert result.best_val_loss < float("inf")