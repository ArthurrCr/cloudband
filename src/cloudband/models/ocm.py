"""OmniCloudMask ensemble: one Dynamic U-Net per backbone, combined by soft voting."""

from __future__ import annotations

from dataclasses import replace
from functools import partial

import timm
import torch
from fastai.data.core import DataLoaders
from fastai.learner import Learner
from fastai.vision.learner import create_unet_model

from cloudband.datasets.cloudsen12 import VALID_SIZE
from cloudband.pipelines.phase0 import RGN_BANDS
from cloudband.train.loop import checkpoint_name, fit_protocol
from cloudband.train.loss import build_loss
from cloudband.train.lr_search import SearchBusy, search_learning_rate
from cloudband.train.manifest import build_training_manifest
from cloudband.train.phase2 import (
    Phase2Run,
    full_protocol,
    load_or_search_learning_rate,
    train_if_missing,
)
from cloudband.train.protocol import TrainProtocol
from cloudband.train.store import RunStore

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


def ocm_search_protocol(protocol: TrainProtocol, backbone_name: str) -> TrainProtocol:
    """The protocol a backbone's learning-rate search runs under.

    Each network of the ensemble has its own search, over the same grid and with the
    same budget as every other network (ADR-0023 D2), kept in its own files. The first
    backbone's search was run before the searches were split by backbone, so it keeps
    the names it was saved under; the others add the backbone to the run id. The
    full runs keep the protocol's own run id whatever the backbone.
    """
    if backbone_name == PAPER_BACKBONES[0]:
        return protocol
    return replace(protocol, run_id=f"{protocol.run_id}-{backbone_name}")


def run_ocm_ensemble_phase2(
    dls: DataLoaders,
    protocol: TrainProtocol,
    seeds: tuple,
    img_size: tuple = (VALID_SIZE, VALID_SIZE),
    pretrained: bool = True,
    search_seed: int | None = None,
) -> dict:
    """Search each backbone's learning rate, then train every backbone once per seed.

    Every network of the ensemble gets its own search, at a single fixed seed (the
    first of seeds by default), over the same grid and with the same budget as the
    other backbones and as Swin, per D2 of ADR-0023: the search is per network, so
    that a backbone is not trained at a rate that was only checked on another one.
    It is run once per backbone and not once per seed, which keeps the seed
    repetitions isolated to initialization variation, rather than mixing in a
    different winning rate for each one. Each backbone then trains at the full budget
    with its own winning rate, once per seed, each with its own checkpoint and manifest.

    Returns a dict keyed by backbone name, each value a dict keyed by seed.
    """
    if search_seed is None:
        search_seed = seeds[0]

    runs: dict = {}
    for backbone_name in PAPER_BACKBONES:
        lr_search = search_learning_rate(
            dls,
            ocm_search_protocol(protocol, backbone_name),
            seed=search_seed,
            model_builder=lambda backbone_name=backbone_name: build_unet(
                backbone_name, img_size=img_size, pretrained=pretrained
            ),
        )
        winning_protocol = full_protocol(protocol, lr_search.winner().learning_rate)
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


def run_ocm_ensemble_resumable(
    dls: DataLoaders,
    protocol: TrainProtocol,
    seeds: tuple,
    store: RunStore,
    img_size: tuple = (VALID_SIZE, VALID_SIZE),
    pretrained: bool = True,
    search_seed: int | None = None,
    progress=print,
    accept_edge_winner: bool = False,
    break_locks: bool = False,
) -> dict[str, str]:
    """Like run_ocm_ensemble_phase2, but saving each run and skipping saved ones.

    Each backbone's learning rate is searched on its own (see ocm_search_protocol)
    and saved. A search that is not finished because another session is running
    some of its candidates is left to it: the runs of that backbone wait, and the
    others go on. Then every seed trains both backbones, one seed at a time, so an
    interruption leaves complete ensembles for the early seeds. Returns each run's
    checkpoint name with "trained", "skipped", "busy" (another session has it) or
    "waiting for its learning-rate search".
    """
    if search_seed is None:
        search_seed = seeds[0]

    chosen: dict[str, tuple] = {}
    for backbone_name in PAPER_BACKBONES:
        try:
            lr_search, _ = load_or_search_learning_rate(
                dls,
                ocm_search_protocol(protocol, backbone_name),
                lambda backbone_name=backbone_name: build_unet(
                    backbone_name, img_size=img_size, pretrained=pretrained
                ),
                store,
                search_seed,
                accept_edge_winner,
                break_locks,
            )
        except SearchBusy as error:
            progress(f"{backbone_name}: its runs wait for the learning-rate search ({error})")
            continue
        winning_protocol = full_protocol(protocol, lr_search.winner().learning_rate)
        chosen[backbone_name] = (lr_search, winning_protocol)
        progress(
            f"{protocol.run_id} {backbone_name}: "
            f"learning rate {winning_protocol.learning_rate}"
        )

    status = {}
    for seed in seeds:
        for backbone_name in PAPER_BACKBONES:
            name = checkpoint_name(protocol, seed, backbone_name)
            if backbone_name not in chosen:
                status[name] = "waiting for its learning-rate search"
                progress(f"{name}: {status[name]}")
                continue
            lr_search, winning_protocol = chosen[backbone_name]
            progress(f"{name}: starting")
            status[name] = train_if_missing(
                dls,
                winning_protocol,
                lr_search,
                seed,
                lambda backbone_name=backbone_name: build_unet(
                    backbone_name, img_size=img_size, pretrained=pretrained
                ),
                store,
                checkpoint_suffix=backbone_name,
                break_locks=break_locks,
            )
            progress(f"{name}: {status[name]}")
    return status
