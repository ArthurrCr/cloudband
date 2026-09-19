"""OmniCloudMask ensemble: one Dynamic U-Net per backbone, combined by soft voting."""

from __future__ import annotations

from functools import partial

import timm
import torch
from fastai.data.core import DataLoaders
from fastai.learner import Learner
from fastai.vision.learner import create_unet_model

from cloudband.datasets.cloudsen12 import VALID_SIZE
from cloudband.pipelines.phase0 import RGN_BANDS
from cloudband.train.loop import fit_protocol
from cloudband.train.loss import build_loss
from cloudband.train.phase2 import Phase2Run, full_protocol
from cloudband.train.lr_search import search_learning_rate
from cloudband.train.manifest import build_training_manifest
from cloudband.train.protocol import TrainProtocol

REGNETY_004 = "regnety_004.pycls_in1k"
CONVNEXTV2_NANO = "convnextv2_nano.fcmae_ft_in1k"
PAPER_BACKBONES = (REGNETY_004, CONVNEXTV2_NANO)

INPUT_CHANNELS = len(RGN_BANDS)
OUTPUT_CLASSES = 4
DECODER_ACTIVATION = torch.nn.Mish


def build_unet(
    backbone_name: str,
    img_size: tuple = (VALID_SIZE, VALID_SIZE),
    pretrained: bool = True,
) -> torch.nn.Module:
    """Build one Dynamic U-Net over a timm backbone.

    img_size only shapes the encoder trace at construction time; the
    resulting model is fully convolutional and accepts any input size at
    runtime, which the mixed-resolution protocol depends on.
    """
    encoder = partial(
        timm.create_model,
        backbone_name,
        pretrained=pretrained,
        in_chans=INPUT_CHANNELS,
    )
    return create_unet_model(
        arch=encoder,
        n_out=OUTPUT_CLASSES,
        img_size=img_size,
        pretrained=pretrained,
        act_cls=DECODER_ACTIVATION,
    )


class OcmEnsemble(torch.nn.Module):
    """Soft-voting ensemble over independently trained backbones.

    Each backbone is trained on its own; this only combines already-trained
    models at inference time.
    """

    def __init__(self, models: tuple):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        params = list(self.parameters())
        device = params[0].device if params else x.device
        x = x.to(device)
        probabilities = [torch.softmax(model(x), dim=1) for model in self.models]
        return torch.stack(probabilities, dim=0).mean(dim=0)


def train_ocm_ensemble(
    dls: DataLoaders,
    protocol: TrainProtocol,
    seed: int,
    img_size: tuple = (VALID_SIZE, VALID_SIZE),
    pretrained: bool = True,
    model_builder=build_unet,
) -> dict:
    """Train one Learner per backbone in the paper's ensemble, same protocol.

    Each backbone gets its own Learner and its own call to fit_protocol; the
    two results are only combined into an OcmEnsemble afterward.
    """
    results = {}
    for backbone_name in PAPER_BACKBONES:
        model = model_builder(backbone_name, img_size=img_size, pretrained=pretrained)
        learner = Learner(dls, model, loss_func=build_loss())
        results[backbone_name] = fit_protocol(
            learner, protocol, seed=seed, checkpoint_suffix=backbone_name
        )
    return results


def run_ocm_ensemble_phase2(
    dls: DataLoaders,
    protocol: TrainProtocol,
    seeds: tuple,
    img_size: tuple = (VALID_SIZE, VALID_SIZE),
    pretrained: bool = True,
    search_seed: int | None = None,
) -> dict:
    """Search the learning rate once, then train every backbone once per seed.

    The learning rate is searched once, at a single fixed seed (the first
    of seeds by default), using the first backbone in PAPER_BACKBONES as
    the representative for "OCM" as an architecture, per D2 of ADR-0023 —
    the ADR partitions the learning-rate search by architecture (OCM vs.
    Swin), not by backbone within OCM's own ensemble, and not by seed:
    searching once keeps the seed repetitions isolated to initialization
    variation, rather than mixing in a different winning rate for each one.
    Every backbone then trains at the full budget with that one winning
    rate, once per seed, each with its own checkpoint and its own manifest.

    Returns a dict keyed by backbone name, each value a dict keyed by seed.
    """
    if search_seed is None:
        search_seed = seeds[0]

    representative = PAPER_BACKBONES[0]
    lr_search = search_learning_rate(
        dls,
        protocol,
        seed=search_seed,
        model_builder=lambda: build_unet(
            representative, img_size=img_size, pretrained=pretrained
        ),
    )
    winner = lr_search.winner()
    winning_protocol = full_protocol(protocol, winner.learning_rate)

    runs: dict = {}
    for backbone_name in PAPER_BACKBONES:
        runs[backbone_name] = {}
        for seed in seeds:
            model = build_unet(backbone_name, img_size=img_size, pretrained=pretrained)
            learner = Learner(dls, model, loss_func=build_loss())
            fit_result = fit_protocol(
                learner, winning_protocol, seed=seed, checkpoint_suffix=backbone_name
            )
            manifest = build_training_manifest(fit_result, winning_protocol)
            runs[backbone_name][seed] = Phase2Run(
                lr_search=lr_search,
                fit_result=fit_result,
                manifest=manifest,
                winning_protocol=winning_protocol,
            )
    return runs