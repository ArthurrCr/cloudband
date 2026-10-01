import math
from functools import partial

import numpy as np
import pytest
import torch

from cloudband.datasets.cloudsen12 import Sample
from cloudband.models import ocm as ocm_module
from cloudband.train import phase2
from cloudband.train.data import build_dataloaders
from cloudband.train.lr_search import search_learning_rate
from cloudband.train.protocol import (
    LR_SEARCH_GRID,
    LrSearchManifest,
    LrSearchRun,
    TrainProtocol,
)
from cloudband.train.store import RunStore

GRID = (3e-5, 1e-4, 3e-4, 1e-3, 3e-3)
PATCH = 32


def search_with_winner(winner, grid=GRID):
    runs = tuple(
        LrSearchRun(f"r{i}", rate, 0.1 if rate == winner else 0.5 + i)
        for i, rate in enumerate(grid)
    )
    return LrSearchManifest(architecture="swin", runs=runs, grid=grid)


def test_the_default_grid_is_five_points_about_half_a_decade_apart():
    steps = [math.log10(b / a) for a, b in zip(LR_SEARCH_GRID, LR_SEARCH_GRID[1:])]

    # 1 and 3 per decade rather than 1 and 3.16, so steps are 0.48 and 0.52
    assert len(LR_SEARCH_GRID) == 5
    assert steps == pytest.approx([0.5] * 4, abs=0.03)


def test_the_grid_contains_the_rate_ocm_trains_with():
    assert 1e-3 in LR_SEARCH_GRID


@pytest.mark.parametrize("winner", [3e-5, 3e-3])
def test_a_winner_on_either_end_is_flagged(winner):
    assert search_with_winner(winner).winner_at_edge() is True


@pytest.mark.parametrize("winner", [1e-4, 3e-4, 1e-3])
def test_a_winner_inside_the_grid_is_not_flagged(winner):
    assert search_with_winner(winner).winner_at_edge() is False


def test_a_grid_without_an_interior_is_never_flagged():
    two_points = search_with_winner(1e-3, grid=(1e-3, 5e-3))

    assert two_points.winner_at_edge() is False


# ---- the training entry point acts on it --------------------------------------


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
    monkeypatch.setattr(phase2, "FULL_FROZEN_EPOCHS", 1)
    monkeypatch.setattr(phase2, "FULL_UNFROZEN_EPOCHS", 1)
    monkeypatch.setattr(
        phase2,
        "search_learning_rate",
        partial(search_learning_rate, frozen_epochs=1, unfrozen_epochs=1, grid=GRID),
    )
    dls = build_dataloaders(
        train_table=range(4),
        valid_table=range(2),
        micro_batch_size=2,
        read_sample=fake_read_sample,
    )
    store = RunStore(tmp_path / "r", tmp_path / "r" / "c", tmp_path / "models")
    protocol = TrainProtocol(
        run_id="swin-rgn-cs12-shared",
        architecture="swin",
        learning_rate=1e-3,
        effective_batch_size=2,
    )
    return dls, store, protocol


def force_winner(monkeypatch, winner):
    """Make the search return a chosen winner without training anything."""
    def fake_search(dls, protocol, seed, model_builder, **kwargs):
        return search_with_winner(winner)

    monkeypatch.setattr(phase2, "search_learning_rate", fake_search)


def test_the_run_stops_before_training_when_the_winner_is_on_an_edge(
    setup, monkeypatch
):
    dls, store, protocol = setup
    force_winner(monkeypatch, 3e-5)
    built = []

    def builder():
        built.append(1)
        return Tiny()

    with pytest.raises(RuntimeError, match="on an end of the grid"):
        phase2.run_phase2_resumable(
            dls, protocol, (0,), builder, store, progress=lambda _: None
        )

    assert built == []                          # not one full run was started
    assert store.load_search(protocol) is not None   # the search itself is kept


def test_the_same_stop_happens_when_the_saved_search_is_reloaded(setup, monkeypatch):
    dls, store, protocol = setup
    store.save_search(protocol, search_with_winner(3e-3))

    with pytest.raises(RuntimeError, match="3e-05|0.003|3e-03"):
        phase2.run_phase2_resumable(
            dls, protocol, (0,), Tiny, store, progress=lambda _: None
        )


def test_accepting_an_edge_winner_lets_training_continue(setup, monkeypatch):
    dls, store, protocol = setup
    force_winner(monkeypatch, 3e-5)

    status = phase2.run_phase2_resumable(
        dls, protocol, (0,), Tiny, store, progress=lambda _: None,
        accept_edge_winner=True,
    )

    assert list(status.values()) == ["trained"]


def test_an_interior_winner_trains_without_any_flag(setup, monkeypatch):
    dls, store, protocol = setup
    force_winner(monkeypatch, 3e-4)

    status = phase2.run_phase2_resumable(
        dls, protocol, (0,), Tiny, store, progress=lambda _: None
    )

    assert list(status.values()) == ["trained"]


def test_the_ocm_entry_point_enforces_the_same_rule(setup, monkeypatch):
    dls, store, _ = setup
    monkeypatch.setattr(ocm_module, "build_unet", lambda *a, **k: Tiny())
    force_winner(monkeypatch, 3e-3)
    protocol = TrainProtocol(
        run_id="ocm-rgn-cs12-shared",
        architecture="ocm",
        learning_rate=1e-3,
        effective_batch_size=2,
    )

    with pytest.raises(RuntimeError, match="on an end of the grid"):
        ocm_module.run_ocm_ensemble_resumable(
            dls, protocol, seeds=(0,), store=store, progress=lambda _: None
        )