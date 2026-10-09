"""Generic Phase 2 orchestration: search the learning rate once, then train
the full run once per seed."""

from __future__ import annotations

import gc
from dataclasses import dataclass, replace

import torch
from fastai.data.core import DataLoaders
from fastai.learner import Learner

from cloudband.provenance.manifest import Manifest
from cloudband.train.loop import FitResult, checkpoint_name, fit_protocol
from cloudband.train.loss import build_loss
from cloudband.train.lr_search import search_learning_rate
from cloudband.train.manifest import build_training_manifest
from cloudband.train.protocol import (
    FULL_FROZEN_EPOCHS,
    FULL_UNFROZEN_EPOCHS,
    LR_SEARCH_GRID,
    LrSearchManifest,
    TrainProtocol,
)
from cloudband.train.store import RunStore


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
    store: RunStore | None = None,
    break_locks: bool = False,
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
        dls,
        protocol,
        seed=search_seed,
        model_builder=model_builder,
        store=store,
        break_locks=break_locks,
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
    checkpoint_suffix: str = "",
    progress_store=None,
) -> Phase2Run:
    """Train one full run at an already-decided winning protocol, for one seed.

    checkpoint_suffix keeps apart models trained under the same run_id and
    seed, such as the backbones of an ensemble.
    """
    learner = Learner(dls, model_builder(), loss_func=build_loss())
    fit_result = fit_protocol(
        learner,
        winning_protocol,
        seed=seed,
        checkpoint_suffix=checkpoint_suffix,
        progress_store=progress_store,
    )
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


def load_or_search_learning_rate(
    dls: DataLoaders,
    protocol: TrainProtocol,
    model_builder,
    store: RunStore,
    search_seed: int,
    accept_edge_winner: bool = False,
    break_locks: bool = False,
) -> tuple[LrSearchManifest, TrainProtocol]:
    """Reuse a saved learning-rate search, or run it once and save it.

    The search is about a hundred epochs, so it must survive a restart: losing
    it would mean repeating it, and repeating it could pick a different winner.

    A winner on either end of the grid means the best rate may lie outside the
    range searched. The hours that follow are not spent on it unless
    accept_edge_winner says so: the protocol is to extend the grid for both
    architectures and search again.
    """
    saved = store.load_search(protocol)
    if saved is not None and tuple(saved.grid) != tuple(LR_SEARCH_GRID):
        # the grid was extended after this search: the candidates already run are
        # kept in the store, so only the new ones are run
        saved = None
    if saved is not None:
        lr_search = saved
        winning_protocol = full_protocol(protocol, saved.winner().learning_rate)
    else:
        lr_search, winning_protocol = search_phase2_learning_rate(
            dls, protocol, model_builder, search_seed, store, break_locks
        )
        store.save_search(protocol, lr_search)

    if lr_search.winner_at_edge() and not accept_edge_winner:
        winner = lr_search.winner().learning_rate
        raise RuntimeError(
            f"{protocol.run_id}: the winning learning rate {winner:g} is on an end "
            f"of the grid {tuple(lr_search.grid)}, so the best rate may lie outside "
            "it. Extend the grid for both architectures and search again (delete "
            f"{store.search_path(protocol).name}; the candidates already run are kept "
            "and not repeated), or pass accept_edge_winner=True to continue and say "
            "so when reporting."
        )
    return lr_search, winning_protocol


def train_if_missing(
    dls: DataLoaders,
    winning_protocol: TrainProtocol,
    lr_search: LrSearchManifest,
    seed: int,
    model_builder,
    store: RunStore,
    checkpoint_suffix: str = "",
    break_locks: bool = False,
) -> str:
    """Train one run unless it is already saved; save it as soon as it ends.

    Returns "skipped" when the run was already on durable storage, "busy" when
    another session is training it, and otherwise "trained". The run saves its
    progress after every epoch, so a session that dies loses at most one epoch: the
    next one continues from the saved state.
    """
    name = checkpoint_name(winning_protocol, seed, checkpoint_suffix)
    if store.is_done(name):
        return "skipped"
    if not store.claim(name, break_locks=break_locks):
        return "busy"

    try:
        run = train_phase2_run(
            dls,
            winning_protocol,
            lr_search,
            seed,
            model_builder,
            checkpoint_suffix,
            progress_store=store,
        )
        store.save_run(run)
        store.clear_epoch_state(name)
    finally:
        store.release(name)
    del run
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return "trained"


def run_phase2_resumable(
    dls: DataLoaders,
    protocol: TrainProtocol,
    seeds: tuple,
    model_builder,
    store: RunStore,
    search_seed: int | None = None,
    progress=print,
    accept_edge_winner: bool = False,
    break_locks: bool = False,
) -> dict[str, str]:
    """Search once, then train one run per seed, saving and skipping as it goes.

    Safe to call again after an interruption: the search and every finished
    seed are read back from the store, and only the missing runs are trained.
    Returns each run's checkpoint name with "trained" or "skipped".
    """
    if search_seed is None:
        search_seed = seeds[0]

    lr_search, winning_protocol = load_or_search_learning_rate(
        dls, protocol, model_builder, store, search_seed, accept_edge_winner,
        break_locks,
    )
    progress(f"{protocol.run_id}: learning rate {winning_protocol.learning_rate}")

    status = {}
    for seed in seeds:
        name = checkpoint_name(winning_protocol, seed)
        progress(f"{name}: starting")
        status[name] = train_if_missing(
            dls, winning_protocol, lr_search, seed, model_builder, store,
            break_locks=break_locks,
        )
        progress(f"{name}: {status[name]}")
    return status