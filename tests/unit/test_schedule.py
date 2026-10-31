"""Every session runs the same cell and takes whatever work is free."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from cloudband.datasets.cloudsen12 import Sample
from cloudband.train import phase2
from cloudband.train.data import build_dataloaders
from cloudband.train.loop import checkpoint_name
from cloudband.train.phase2 import full_protocol
from cloudband.train.protocol import LrSearchRun, ocm_shared_protocol
from cloudband.train.schedule import Network, run_everything
from cloudband.train.store import RunStore

GRID = (1e-3, 3e-3)
SEEDS = (0, 1)
BUDGET = (5, 5)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=1)

    def forward(self, x):
        return self.conv(x.float())


def fake_read_sample(table, index, crop_to_valid=True):
    rng = np.random.default_rng(index)
    return Sample(
        identifier=f"scene-{index}",
        image=rng.integers(0, 10000, (13, 32, 32)).astype(np.int32),
        annotation=rng.integers(0, 4, (32, 32)).astype(np.int32),
    )


def base_protocol():
    return replace(
        ocm_shared_protocol(1e-3),
        seeds=SEEDS,
        effective_batch_size=8,
        min_gsd_m=300.0,
        max_gsd_m=400.0,
        mixed_precision=False,
    )


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(phase2, "FULL_FROZEN_EPOCHS", 1)
    monkeypatch.setattr(phase2, "FULL_UNFROZEN_EPOCHS", 1)
    base = base_protocol()
    dls = build_dataloaders(train_table=range(8), valid_table=range(4), micro_batch_size=4,
                            read_sample=fake_read_sample, augmentation=base.augmentation)
    networks = [
        Network("A", base, base, Tiny, dls, suffix="a"),
        Network("B", replace(base, run_id=f"{base.run_id}-b"), base, Tiny, dls, suffix="b"),
    ]
    store = RunStore(tmp_path / "drive" / "results", tmp_path / "drive" / "ckpt",
                     tmp_path / "models")
    return networks, store


def run_name(network, rate, seed):
    return checkpoint_name(full_protocol(network.base_protocol, rate), seed, network.suffix)


def winner(store, network):
    return store.load_search(network.search_protocol).winner().learning_rate


def test_one_session_does_every_search_then_every_run(setup):
    networks, store = setup

    outcome = run_everything(networks, store, SEEDS, grid=GRID, wait=False, say=lambda _m: None)

    assert outcome.finished
    assert outcome.trained == [
        "A candidate 0.001", "B candidate 0.001", "A candidate 0.003", "B candidate 0.003",
        run_name(networks[0], winner(store, networks[0]), 0),
        run_name(networks[1], winner(store, networks[1]), 0),
        run_name(networks[0], winner(store, networks[0]), 1),
        run_name(networks[1], winner(store, networks[1]), 1),
    ]
    for network in networks:
        for seed in SEEDS:
            assert store.is_done(run_name(network, winner(store, network), seed))

    again = run_everything(networks, store, SEEDS, grid=GRID, wait=False, say=lambda _m: None)
    assert again.trained == [] and again.finished


def test_a_session_works_around_what_another_one_holds_and_waits_for_a_search(setup):
    networks, store = setup
    other = RunStore(store.results_dir, store.checkpoint_dir, store.source_dir)
    held = checkpoint_name(
        replace(networks[1].search_protocol, learning_rate=3e-3), 0, "lrsearch-0.003"
    )
    assert other.claim(held, sleep=lambda _s: None)

    waits = []

    def other_session_finishes(seconds):
        waits.append(seconds)
        other.save_search_run(
            networks[1].search_protocol,
            LrSearchRun(networks[1].search_protocol.run_id, 3e-3, 9.9),
            BUDGET,
            0,
        )
        other.release(held)

    outcome = run_everything(networks, store, SEEDS, grid=GRID, sleep=other_session_finishes,
                             poll_seconds=60, say=lambda _m: None)

    assert waits == [60]
    assert "B candidate 0.003" not in outcome.trained
    assert outcome.finished
    assert winner(store, networks[1]) == 1e-3
    for seed in SEEDS:
        assert store.is_done(run_name(networks[1], 1e-3, seed))


def test_without_waiting_the_session_reports_what_is_left(setup):
    networks, store = setup
    other = RunStore(store.results_dir, store.checkpoint_dir, store.source_dir)
    held = checkpoint_name(
        replace(networks[1].search_protocol, learning_rate=3e-3), 0, "lrsearch-0.003"
    )
    other.claim(held, sleep=lambda _s: None)

    outcome = run_everything(networks, store, SEEDS, grid=GRID, wait=False, say=lambda _m: None)

    assert outcome.held_elsewhere == ["B candidate 0.003"]
    assert outcome.waiting_for_search == ["B seed 0", "B seed 1"]
    for seed in SEEDS:
        assert store.is_done(run_name(networks[0], winner(store, networks[0]), seed))


def test_a_winner_on_the_edge_of_the_grid_stops_that_network_only(setup):
    networks, store = setup
    grid = (1e-3, 3e-3, 1e-2)
    for network in networks:
        losses = (0.5, 0.4, 0.3) if network.label == "A" else (0.5, 0.3, 0.4)
        for rate, loss in zip(grid, losses):
            store.save_search_run(
                network.search_protocol,
                LrSearchRun(network.search_protocol.run_id, rate, loss),
                BUDGET,
                0,
            )
    messages = []

    outcome = run_everything(networks, store, SEEDS, grid=grid, wait=False, say=messages.append)

    assert len(outcome.at_grid_edge) == 1 and outcome.at_grid_edge[0].startswith("A:")
    assert not any(name.endswith("-a") for name in outcome.trained)
    assert all(store.is_done(run_name(networks[1], 3e-3, seed)) for seed in SEEDS)
    assert any("grid must be extended" in message for message in messages)
