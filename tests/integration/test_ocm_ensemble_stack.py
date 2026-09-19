import numpy as np
import torch

from cloudband.datasets.cloudsen12 import Sample
from cloudband.models.ocm import PAPER_BACKBONES, OcmEnsemble, train_ocm_ensemble
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


class TinyBackboneStandIn(torch.nn.Module):
    """A cheap stand-in for a real timm-backed Dynamic U-Net.

    Real backbone construction and training are verified separately, once
    per backbone, in test_ocm_model.py and by direct manual run; this stands
    in only to keep the orchestration test (both backbones, both results
    collected, ensemble assembled) light enough for the sandbox.
    """

    def __init__(self, backbone_name, img_size, pretrained):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=1)

    def forward(self, x):
        return self.conv(x.float())


def fake_model_builder(backbone_name, img_size, pretrained):
    return TinyBackboneStandIn(backbone_name, img_size, pretrained)


def test_train_ocm_ensemble_trains_both_backbones_end_to_end():
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
        frozen_epochs=1,
        unfrozen_epochs=1,
        effective_batch_size=2,
    )

    results = train_ocm_ensemble(
        dls,
        protocol,
        seed=protocol.seeds[0],
        pretrained=False,
        model_builder=fake_model_builder,
    )

    assert set(results) == set(PAPER_BACKBONES)
    for result in results.values():
        assert result.learner.recorder.losses
        assert result.best_val_loss < float("inf")


def test_ensemble_from_trained_learners_produces_valid_probabilities():
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
        frozen_epochs=1,
        unfrozen_epochs=1,
        effective_batch_size=2,
    )

    results = train_ocm_ensemble(
        dls,
        protocol,
        seed=protocol.seeds[0],
        pretrained=False,
        model_builder=fake_model_builder,
    )
    ensemble = OcmEnsemble(tuple(result.learner.model for result in results.values()))
    ensemble.eval()

    images, _ = next(iter(dls.valid))
    probabilities = ensemble(images.float())

    assert probabilities.shape == (images.shape[0], 4, PATCH_SIZE, PATCH_SIZE)
    totals = probabilities.sum(dim=1)
    assert (totals - 1.0).abs().max() < 1e-4