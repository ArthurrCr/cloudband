"""Learning-rate search over the shared grid, per ADR-0023 D2."""

from __future__ import annotations

import gc
from dataclasses import replace

import torch
from fastai.data.core import DataLoaders
from fastai.learner import Learner

from cloudband.train.loop import TrainingDiverged, checkpoint_name, fit_protocol
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
    store=None,
    break_locks: bool = False,
) -> LrSearchManifest:
    """Run one proxy training per candidate learning rate and record all of them.

    model_builder takes no arguments and returns a fresh, untrained model;
    every candidate starts from scratch, never from a previous candidate's
    weights. dls carries only train and validation data, so the search
    structurally cannot touch the test split: there is no parameter here it
    could come in through.

    With a store, a finished candidate is kept the moment it ends and is not run
    again, and a candidate in flight saves its progress after every epoch. A
    candidate that another session is running is left to it: the search then
    stops with an error naming the candidates still missing, to be run again once
    that session is done. Each candidate has its own checkpoint name, so nothing
    one candidate saves can be taken for another's.
    """
    runs: list[LrSearchRun] = []
    busy: list[float] = []
    budget = (frozen_epochs, unfrozen_epochs)
    for learning_rate in grid:
        saved = (
            store.load_search_run(protocol, learning_rate, budget, seed)
            if store
            else None
        )
        if saved is not None:
            runs.append(saved)
            continue

        candidate = proxy_protocol(
            protocol, learning_rate, frozen_epochs, unfrozen_epochs
        )
        suffix = f"lrsearch-{learning_rate:g}"
        name = checkpoint_name(candidate, seed, suffix)
        if store and not store.claim(name, break_locks=break_locks):
            busy.append(learning_rate)
            continue
        try:
            learner = Learner(dls, model_builder(), loss_func=build_loss())
            try:
                result = fit_protocol(
                    learner, candidate, seed=seed, checkpoint_suffix=suffix,
                    progress_store=store,
                )
                val_loss = result.best_val_loss
            except TrainingDiverged as error:
                # a rate that blows up is a result of the search, the worst one,
                # not a reason to stop it
                print(f"{name}: diverged, recorded as infinite loss ({error})")
                val_loss = float("inf")
                result = None
            run = LrSearchRun(
                run_id=candidate.run_id,
                learning_rate=learning_rate,
                val_loss=val_loss,
            )
            if store:
                store.save_search_run(protocol, run, budget, seed)
                store.clear_epoch_state(name)
            runs.append(run)
            del learner, result
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            if store:
                store.release(name)

    if busy:
        raise RuntimeError(
            f"{protocol.run_id}: the candidates {busy} are being run by another "
            "session. Run this cell again when it has finished them, or pass "
            "break_locks=True if that session is gone."
        )
    return LrSearchManifest(
        architecture=protocol.architecture, runs=tuple(runs), grid=grid
    )