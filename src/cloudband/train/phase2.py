"""Generic Phase 2 orchestration: search the learning rate once, then train
the full run once per seed."""

from __future__ import annotations

from dataclasses import dataclass, replace

from fastai.data.core import DataLoaders
from fastai.learner import Learner

from cloudband.provenance.manifest import Manifest
from cloudband.train.loop import FitResult, fit_protocol
from cloudband.train.loss import build_loss
from cloudband.train.lr_search import search_learning_rate
from cloudband.train.manifest import build_training_manifest
from cloudband.train.protocol import (
    FULL_FROZEN_EPOCHS,
    FULL_UNFROZEN_EPOCHS,
    LrSearchManifest,
    TrainProtocol,
)


@dataclass(frozen=True)
class Phase2Run:
    """Everything one seed's full training run produces.

    lr_search is the same manifest shared by every seed under one
    architecture: the search runs once, not once per seed.
    """

    lr_search: LrSearchManifest
    fit_result: FitResult
    manifest: Manifest
    winning_protocol: TrainProtocol


def full_protocol(protocol: TrainProtocol, learning_rate: float) -> TrainProtocol:
    """Expand a protocol to the full training budget, at the winning learning rate."""
    return replace(
        protocol,
        learning_rate=learning_rate,
        frozen_epochs=FULL_FROZEN_EPOCHS,
        unfrozen_epochs=FULL_UNFROZEN_EPOCHS,
    )


def search_phase2_learning_rate(
    dls: DataLoaders,
    protocol: TrainProtocol,
    model_builder,
    search_seed: int,
) -> tuple[LrSearchManifest, TrainProtocol]:
    """Search the learning rate once, returning the manifest and the winning
    full-budget protocol.

    Runs once, at a single fixed seed, separate from the seeds used to
    repeat the final training. Mixing the two would let a different winning
    rate get picked by chance for each seed, entangling two sources of
    variation that need to stay apart: how the model responds to its
    initialization, and which learning rate the search happened to prefer.
    """
    lr_search = search_learning_rate(
        dls, protocol, seed=search_seed, model_builder=model_builder
    )
    winner = lr_search.winner()
    winning_protocol = full_protocol(protocol, winner.learning_rate)
    return lr_search, winning_protocol


def train_phase2_run(
    dls: DataLoaders,
    winning_protocol: TrainProtocol,
    lr_search: LrSearchManifest,
    seed: int,
    model_builder,
) -> Phase2Run:
    """Train one full run at an already-decided winning protocol, for one seed."""
    learner = Learner(dls, model_builder(), loss_func=build_loss())
    fit_result = fit_protocol(learner, winning_protocol, seed=seed)
    manifest = build_training_manifest(fit_result, winning_protocol)
    return Phase2Run(
        lr_search=lr_search,
        fit_result=fit_result,
        manifest=manifest,
        winning_protocol=winning_protocol,
    )


def run_phase2(
    dls: DataLoaders,
    protocol: TrainProtocol,
    seeds: tuple,
    model_builder,
    search_seed: int | None = None,
) -> tuple[Phase2Run, ...]:
    """Search the learning rate once, then train the full run once per seed.

    model_builder takes no arguments and returns a fresh, untrained model.
    It is called once per candidate during the search and once more per
    seed for the full runs, so every run starts from scratch, never from a
    previous run's weights. search_seed defaults to the first of seeds.
    """
    if search_seed is None:
        search_seed = seeds[0]

    lr_search, winning_protocol = search_phase2_learning_rate(
        dls, protocol, model_builder, search_seed
    )
    return tuple(
        train_phase2_run(dls, winning_protocol, lr_search, seed, model_builder)
        for seed in seeds
    )