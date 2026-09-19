"""Generic Phase 2 orchestration: search the learning rate, train the winner,
record it."""

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
    """Everything one architecture's Phase 2 run produces."""

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


def run_phase2(
    dls: DataLoaders,
    protocol: TrainProtocol,
    seed: int,
    model_builder,
) -> Phase2Run:
    """Search the learning rate, train the full run at the winner, record it.

    model_builder takes no arguments and returns a fresh, untrained model.
    It is called once per candidate during the search and once more for the
    full run, so every run starts from scratch, never from a previous
    candidate's weights.
    """
    lr_search = search_learning_rate(
        dls, protocol, seed=seed, model_builder=model_builder
    )
    winner = lr_search.winner()
    winning_protocol = full_protocol(protocol, winner.learning_rate)

    learner = Learner(dls, model_builder(), loss_func=build_loss())
    fit_result = fit_protocol(learner, winning_protocol, seed=seed)
    manifest = build_training_manifest(fit_result, winning_protocol)

    return Phase2Run(
        lr_search=lr_search,
        fit_result=fit_result,
        manifest=manifest,
        winning_protocol=winning_protocol,
    )