from dataclasses import replace

import numpy as np
import pytest
import torch

from cloudband.datasets.cloudsen12 import Sample
from cloudband.train import phase2
from cloudband.train.data import build_dataloaders
from cloudband.train.loop import checkpoint_name
from cloudband.train.lr_search import search_learning_rate
from cloudband.train.phase2 import full_protocol, train_if_missing
from cloudband.train.protocol import LrSearchManifest, LrSearchRun, ocm_shared_protocol
from cloudband.train.store import RunStore

FROZEN, UNFROZEN = 2, 2


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Conv2d(3, 8, kernel_size=1)
        self.second = torch.nn.Conv2d(8, 4, kernel_size=1)

    def forward(self, x):
        return self.second(torch.relu(self.first(x.float())))


class SessionLost(RuntimeError):
    pass


def seeded(model_class):
    """A builder that always starts from the same initial weights.

    Two runs that are to be compared must start from the same model, and the
    initialisation draws from the global generator, which earlier work has moved.
    """

    def build():
        torch.manual_seed(11)
        return model_class()

    return build


def dying_after(forward_calls):
    """A model builder whose model dies once it has run `forward_calls` batches."""

    class Dying(Tiny):
        calls = 0

        def forward(self, x):
            type(self).calls += 1
            if type(self).calls > forward_calls:
                raise SessionLost("the session was lost")
            return super().forward(x)

    return seeded(Dying)


def fake_read_sample(table, index, crop_to_valid=True):
    rng = np.random.default_rng(index)
    return Sample(
        identifier=f"scene-{index}",
        image=rng.integers(0, 10000, (13, 32, 32)).astype(np.int32),
        annotation=rng.integers(0, 4, (32, 32)).astype(np.int32),
    )


def protocol():
    return replace(
        ocm_shared_protocol(1e-3),
        effective_batch_size=8,
        min_gsd_m=300.0,
        max_gsd_m=400.0,
        mixed_precision=False,
    )


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(phase2, "FULL_FROZEN_EPOCHS", FROZEN)
    monkeypatch.setattr(phase2, "FULL_UNFROZEN_EPOCHS", UNFROZEN)
    dls = build_dataloaders(train_table=range(16), valid_table=range(8),
                            micro_batch_size=4, read_sample=fake_read_sample)
    store = RunStore(
        tmp_path / "drive" / "results", tmp_path / "drive" / "ckpt", tmp_path / "models"
    )
    return dls, store


def winning(lr=1e-3):
    base = protocol()
    search = LrSearchManifest("ocm", (LrSearchRun("r", lr, 0.4),))
    return full_protocol(base, lr), search


# one epoch is 4 training batches and 2 validation batches
CALLS_PER_EPOCH = 6


def test_a_run_that_dies_is_continued_by_the_next_session(setup):
    dls, store = setup
    run_protocol, search = winning()
    name = checkpoint_name(run_protocol, 0)

    with pytest.raises(SessionLost):
        train_if_missing(dls, run_protocol, search, 0,
                         dying_after(CALLS_PER_EPOCH * 3 + 2), store)

    assert not store.is_done(name)
    saved = store.load_epoch_state(name)
    assert (saved["phase"], saved["epochs_done"]) == ("unfrozen", 1)   # 3 epochs done
    assert store.lock_info(name) is None                  # the dead session let go

    resumed = []

    def builder():
        model = Tiny()
        resumed.append(model)
        return model

    assert train_if_missing(dls, run_protocol, search, 0, builder, store) == "trained"
    assert store.is_done(name)
    assert store.load_epoch_state(name) is None           # progress cleared once saved
    assert store.convergence(name)["epochs"] == FROZEN + UNFROZEN


def test_the_continued_run_matches_a_run_that_never_died(setup, tmp_path):
    dls, store = setup
    run_protocol, search = winning()
    name = checkpoint_name(run_protocol, 0)

    with pytest.raises(SessionLost):
        train_if_missing(dls, run_protocol, search, 0,
                         dying_after(CALLS_PER_EPOCH * 2 + 3), store)
    train_if_missing(dls, run_protocol, search, 0, seeded(Tiny), store)
    interrupted = torch.load(store.checkpoint_path(name), weights_only=False)

    # fastai writes checkpoints under ./models of the working directory
    clean_store = RunStore(tmp_path / "clean" / "r", tmp_path / "clean" / "c",
                           tmp_path / "models")
    train_if_missing(dls, run_protocol, search, 0, seeded(Tiny), clean_store)
    clean = torch.load(clean_store.checkpoint_path(name), weights_only=False)

    assert all(torch.equal(interrupted[k], clean[k]) for k in clean)


def test_a_run_another_session_is_training_is_left_alone(setup):
    dls, store = setup
    run_protocol, search = winning()
    name = checkpoint_name(run_protocol, 0)
    rival = RunStore(store.results_dir, store.checkpoint_dir, store.source_dir)
    rival.claim(name, sleep=lambda s: None)
    built = []

    status = train_if_missing(dls, run_protocol, search, 0,
                              lambda: built.append(1) or Tiny(), store)

    assert status == "busy"
    assert built == []


def test_a_dead_session_can_be_taken_over_explicitly(setup):
    dls, store = setup
    run_protocol, search = winning()
    name = checkpoint_name(run_protocol, 0)
    rival = RunStore(store.results_dir, store.checkpoint_dir, store.source_dir)
    rival.claim(name, sleep=lambda s: None)

    status = train_if_missing(dls, run_protocol, search, 0, Tiny, store,
                              break_locks=True)

    assert status == "trained"


# ---- the learning-rate search ------------------------------------------------------

GRID = (1e-3, 3e-3, 1e-2)


def run_search(dls, store, builder, **kwargs):
    return search_learning_rate(
        dls, protocol(), seed=0, model_builder=builder, grid=GRID,
        frozen_epochs=1, unfrozen_epochs=1, store=store, **kwargs,
    )


def test_a_search_that_dies_keeps_its_finished_candidates(setup):
    dls, store = setup
    # candidate one is two epochs, 12 batches; the model dies inside candidate two
    with pytest.raises(SessionLost):
        run_search(dls, store, dying_after(CALLS_PER_EPOCH * 2 + CALLS_PER_EPOCH + 2))

    kept = store.load_search_run(protocol(), 1e-3, (1, 1), 0)
    assert kept is not None
    assert store.load_search_run(protocol(), 3e-3, (1, 1), 0) is None


def test_the_next_session_finishes_the_search_without_repeating_candidates(setup):
    dls, store = setup
    with pytest.raises(SessionLost):
        run_search(dls, store, dying_after(CALLS_PER_EPOCH * 2 + CALLS_PER_EPOCH + 2))
    built = []

    def counting():
        built.append(1)
        return Tiny()

    manifest = run_search(dls, store, counting)

    assert [r.learning_rate for r in manifest.runs] == list(GRID)
    assert len(built) == 2        # candidate one was not built again: two and three


def test_a_search_in_two_pieces_matches_one_that_never_stopped(setup, tmp_path):
    dls, store = setup
    with pytest.raises(SessionLost):
        run_search(dls, store, dying_after(CALLS_PER_EPOCH * 2 + CALLS_PER_EPOCH + 2))
    pieces = run_search(dls, store, seeded(Tiny))

    clean = RunStore(
        tmp_path / "c" / "r", tmp_path / "c" / "k", tmp_path / "models"
    )
    whole = run_search(dls, clean, seeded(Tiny))

    assert [r.val_loss for r in pieces.runs] == [r.val_loss for r in whole.runs]
    assert pieces.winner().learning_rate == whole.winner().learning_rate


def test_a_candidate_another_session_runs_stops_the_search_with_a_clear_error(setup):
    dls, store = setup
    rival = RunStore(store.results_dir, store.checkpoint_dir, store.source_dir)
    name = checkpoint_name(replace(protocol(), frozen_epochs=1, unfrozen_epochs=1),
                           0, "lrsearch-0.003")
    rival.claim(name, sleep=lambda s: None)

    with pytest.raises(RuntimeError, match=r"candidates \[0.003\] are being run"):
        run_search(dls, store, Tiny)

    # the other two candidates were still done, and are kept
    assert store.load_search_run(protocol(), 1e-3, (1, 1), 0) is not None
    assert store.load_search_run(protocol(), 1e-2, (1, 1), 0) is not None