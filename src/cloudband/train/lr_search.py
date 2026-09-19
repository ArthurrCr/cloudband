"""Learning-rate search over the shared grid, per ADR-0023 D2."""

from __future__ import annotations

from dataclasses import replace

from fastai.data.core import DataLoaders
from fastai.learner import Learner

from cloudband.train.loop import fit_protocol
from cloudband.train.loss import build_loss
from cloudband.train.protocol import (
    LR_SEARCH_GRID,
    PROXY_FROZEN_EPOCHS,
    PROXY_UNFROZEN_EPOCHS,
    LrSearchManifest,
    LrSearchRun,
    TrainProtocol,
)


def proxy_protocol(
    protocol: TrainProtocol,
    learning_rate: float,
    frozen_epochs: int = PROXY_FROZEN_EPOCHS,
    unfrozen_epochs: int = PROXY_UNFROZEN_EPOCHS,
) -> TrainProtocol:
    """Shorten a protocol to the proxy-run budget, at a candidate learning rate."""
    return replace(
        protocol,
        learning_rate=learning_rate,
        frozen_epochs=frozen_epochs,
        unfrozen_epochs=unfrozen_epochs,
    )


def search_learning_rate(
    dls: DataLoaders,
    protocol: TrainProtocol,
    seed: int,
    model_builder,
    grid: tuple = LR_SEARCH_GRID,
    frozen_epochs: int = PROXY_FROZEN_EPOCHS,
    unfrozen_epochs: int = PROXY_UNFROZEN_EPOCHS,
) -> LrSearchManifest:
    """Run one proxy training per candidate learning rate and record all of them.

    model_builder takes no arguments and returns a fresh, untrained model;
    every candidate starts from scratch, never from a previous candidate's
    weights. dls carries only train and validation data, so the search
    structurally cannot touch the test split: there is no parameter here it
    could come in through.
    """
    runs = []
    for learning_rate in grid:
        candidate = proxy_protocol(
            protocol, learning_rate, frozen_epochs, unfrozen_epochs
        )
        model = model_builder()
        learner = Learner(dls, model, loss_func=build_loss())
        result = fit_protocol(learner, candidate, seed=seed)
        runs.append(
            LrSearchRun(
                run_id=candidate.run_id,
                learning_rate=learning_rate,
                val_loss=result.best_val_loss,
            )
        )
    return LrSearchManifest(
        architecture=protocol.architecture, runs=tuple(runs), grid=grid
    )