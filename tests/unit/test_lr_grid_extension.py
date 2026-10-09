"""Extending the learning-rate grid after a search, and a candidate that blows up."""

from functools import partial

import numpy as np
import pytest
import torch

from cloudband.datasets.cloudsen12 import Sample
from cloudband.train import lr_search as lr_search_module
from cloudband.train import phase2
from cloudband.train.data import build_dataloaders
from cloudband.train.loop import TrainingDiverged
from cloudband.train.protocol import (
    LR_SEARCH_GRID,
    LrSearchManifest,
    LrSearchRun,
    TrainProtocol,
)
from cloudband.train.store import RunStore

PATCH = 32
OLD_GRID = (3e-5, 1e-4, 3e-4, 1e-3, 3e-3)
NEW_RATE = 1e-2


def fake_read_sample(table, index, crop_to_valid=True):
    rng = np.random.default_rng(index)
    return Sample(
        identifier=f"scene-{index}",
        image=rng.integers(0, 10000, (13, PATCH, PATCH)).astype(np.int32),
        annotation=rng.integers(0, 4, (PATCH, PATCH)).astype(np.int32),
    )


class Tiny(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=1)

    def forward(self, x):
        return self.conv(x.float())


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dls = build_dataloaders(
        train_table=range(4),
        valid_table=range(2),
        micro_batch_size=2,
        read_sample=fake_read_sample,
    )
    store = RunStore(tmp_path / "r", tmp_path / "r" / "c", tmp_path / "models")
    protocol = TrainProtocol(
        run_id="ocm-rgn-cs12-shared",
        architecture="ocm",
        learning_rate=1e-3,
        effective_batch_size=2,
    )
    return dls, store, protocol


def spy_on_fits(monkeypatch, diverge_at=None):
    """Record the rate of every candidate trained; make one of them diverge."""
    real_fit = lr_search_module.fit_protocol
    trained = []

    def fit(learner, candidate, **kwargs):
        trained.append(candidate.learning_rate)
        if candidate.learning_rate == diverge_at:
            raise TrainingDiverged(f"{candidate.run_id}: loss was NaN")
        return real_fit(learner, candidate, **kwargs)

    monkeypatch.setattr(lr_search_module, "fit_protocol", fit)
    return trained


def test_the_grid_was_extended_one_step_above_the_old_one():
    assert LR_SEARCH_GRID == (*OLD_GRID, NEW_RATE)


def test_a_candidate_that_diverges_is_recorded_as_the_worst_and_the_search_goes_on(
    setup, monkeypatch
):
    dls, store, protocol = setup
    trained = spy_on_fits(monkeypatch, diverge_at=NEW_RATE)

    search = lr_search_module.search_learning_rate(
        dls, protocol, seed=0, model_builder=Tiny, store=store,
        frozen_epochs=1, unfrozen_epochs=1,
    )

    by_rate = {run.learning_rate: run.val_loss for run in search.runs}
    assert trained == list(LR_SEARCH_GRID)
    assert by_rate[NEW_RATE] == float("inf")
    assert search.winner().learning_rate != NEW_RATE
    # kept like any other candidate: a second search does not run it again
    again = spy_on_fits(monkeypatch)
    lr_search_module.search_learning_rate(
        dls, protocol, seed=0, model_builder=Tiny, store=store,
        frozen_epochs=1, unfrozen_epochs=1,
    )
    assert again == []


def test_a_search_saved_on_the_old_grid_runs_only_the_new_candidate(setup, monkeypatch):
    dls, store, protocol = setup
    for index, rate in enumerate(OLD_GRID):
        store.save_search_run(
            protocol,
            LrSearchRun(f"r{index}", rate, 0.5 - 0.05 * index),
            (1, 1),
            0,
        )
    store.save_search(
        protocol,
        LrSearchManifest(
            architecture="ocm",
            runs=tuple(
                LrSearchRun(f"r{i}", rate, 0.5 - 0.05 * i)
                for i, rate in enumerate(OLD_GRID)
            ),
            grid=OLD_GRID,
        ),
    )
    monkeypatch.setattr(
        phase2,
        "search_learning_rate",
        partial(lr_search_module.search_learning_rate, frozen_epochs=1, unfrozen_epochs=1),
    )
    trained = spy_on_fits(monkeypatch)

    search, _ = phase2.load_or_search_learning_rate(
        dls, protocol, Tiny, store, search_seed=0, accept_edge_winner=True
    )

    assert trained == [NEW_RATE]
    assert tuple(search.grid) == LR_SEARCH_GRID
    assert len(search.runs) == len(LR_SEARCH_GRID)
    assert tuple(store.load_search(protocol).grid) == LR_SEARCH_GRID


def test_a_search_on_the_current_grid_is_reused_without_training(setup, monkeypatch):
    dls, store, protocol = setup
    runs = tuple(
        LrSearchRun(f"r{i}", rate, 0.9 - 0.1 * i) for i, rate in enumerate(LR_SEARCH_GRID)
    )
    store.save_search(
        protocol, LrSearchManifest(architecture="ocm", runs=runs, grid=LR_SEARCH_GRID)
    )
    trained = spy_on_fits(monkeypatch)

    search, winning = phase2.load_or_search_learning_rate(
        dls, protocol, Tiny, store, search_seed=0, accept_edge_winner=True
    )

    assert trained == []
    assert winning.learning_rate == NEW_RATE
    assert search.winner().learning_rate == NEW_RATE
