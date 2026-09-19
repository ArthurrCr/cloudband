import numpy as np
import torch

from cloudband.datasets.cloudsen12 import Sample
from cloudband.train.data import build_dataloaders
from cloudband.train.loop import fit_protocol
from cloudband.train.protocol import TrainProtocol

PATCH_SIZE = 509


def fake_read_sample(table, index, crop_to_valid=True):
    rng = np.random.default_rng(index)
    image = rng.integers(0, 10000, size=(13, PATCH_SIZE, PATCH_SIZE))
    annotation = rng.integers(0, 4, size=(PATCH_SIZE, PATCH_SIZE))
    return Sample(
        identifier=f"scene-{index}",
        image=image.astype(np.int32),
        annotation=annotation.astype(np.int32),
    )


class TinySegmenter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=1)

    def forward(self, x):
        return self.conv(x.float())


def test_full_stack_from_fake_samples_through_a_real_training_step():
    from fastai.learner import Learner
    from fastai.losses import CrossEntropyLossFlat

    dls = build_dataloaders(
        train_table=range(4),
        valid_table=range(4),
        micro_batch_size=2,
        read_sample=fake_read_sample,
    )
    learner = Learner(dls, TinySegmenter(), loss_func=CrossEntropyLossFlat(axis=1))
    protocol = TrainProtocol(
        run_id="ocm-rgn-cs12-shared",
        architecture="ocm",
        learning_rate=1e-3,
        frozen_epochs=1,
        unfrozen_epochs=1,
        effective_batch_size=4,
    )

    fit_protocol(learner, protocol, seed=protocol.seeds[0])

    assert learner.recorder.losses