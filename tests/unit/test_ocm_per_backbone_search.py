"""Each network of the OCM ensemble has its own learning-rate search (ADR-0023 D2)."""

import json
from dataclasses import replace

import pytest

from cloudband.models import ocm as ocm_module
from cloudband.models.ocm import PAPER_BACKBONES, ocm_search_protocol
from cloudband.train.loop import checkpoint_name
from cloudband.train.lr_search import SearchBusy
from cloudband.train.phase2 import train_if_missing
from cloudband.train.protocol import (
    LrSearchManifest,
    LrSearchRun,
    ocm_shared_protocol,
)
from cloudband.train.status import Plan, status_report
from cloudband.train.store import RunStore

REGNETY, CONVNEXT = PAPER_BACKBONES


def make_store(tmp_path):
    return RunStore(
        results_dir=tmp_path / "results",
        checkpoint_dir=tmp_path / "checkpoints",
        source_dir=tmp_path / "models",
    )


def search_won_by(rate):
    return LrSearchManifest("ocm", (LrSearchRun("r", rate, 0.4),))


def test_the_first_backbone_keeps_the_names_its_search_was_saved_under():
    protocol = ocm_shared_protocol(1e-3)

    assert ocm_search_protocol(protocol, REGNETY) == protocol


def test_the_other_backbone_searches_under_its_own_run_id():
    protocol = ocm_shared_protocol(1e-3)

    other = ocm_search_protocol(protocol, CONVNEXT)

    assert other.run_id == f"{protocol.run_id}-{CONVNEXT}"
    assert replace(other, run_id=protocol.run_id) == protocol


def test_discard_removes_everything_a_run_left(tmp_path):
    store = make_store(tmp_path)
    name = "some-run"
    store.checkpoint_path(name).write_bytes(b"x")
    store.manifest_path(name).write_text("{}")
    store.save_epoch_state(name, {"valid_losses": [1.0]})
    assert store.claim(name, sleep=lambda _: None)
    store.save_epoch_state("other-run", {"valid_losses": [1.0]})

    removed = store.discard(name)

    assert len(removed) == 4
    assert not store.is_done(name)
    assert store.load_epoch_state(name) is None
    assert store.lock_info(name) is None
    assert store.load_epoch_state("other-run") is not None


def test_a_finished_run_at_another_rate_than_the_chosen_one_is_not_kept(tmp_path):
    store = make_store(tmp_path)
    chosen = replace(ocm_shared_protocol(1e-3), learning_rate=1e-3)
    name = checkpoint_name(chosen, 0, CONVNEXT)
    store.checkpoint_path(name).write_bytes(b"x")
    store.manifest_path(name).write_text(json.dumps({"config": {"learning_rate": 3e-3}}))

    with pytest.raises(RuntimeError, match="discard it"):
        train_if_missing(
            None, chosen, search_won_by(1e-3), 0, None, store, checkpoint_suffix=CONVNEXT
        )

    store.discard(name)
    assert not store.is_done(name)


def test_a_finished_run_at_the_chosen_rate_is_skipped(tmp_path):
    store = make_store(tmp_path)
    chosen = ocm_shared_protocol(1e-3)
    name = checkpoint_name(chosen, 0, CONVNEXT)
    store.checkpoint_path(name).write_bytes(b"x")
    store.manifest_path(name).write_text(json.dumps({"config": {"learning_rate": 1e-3}}))

    result = train_if_missing(
        None, chosen, search_won_by(1e-3), 0, None, store, checkpoint_suffix=CONVNEXT
    )

    assert result == "skipped"


def test_the_status_flags_a_run_trained_at_another_rate_than_its_search_chose(tmp_path):
    store = make_store(tmp_path)
    protocol = replace(ocm_shared_protocol(1e-3), seeds=(0,))
    plan = Plan(f"OCM {CONVNEXT}", protocol, CONVNEXT, ocm_search_protocol(protocol, CONVNEXT))
    store.save_search(
        plan.search,
        LrSearchManifest(
            architecture="ocm",
            runs=(
                LrSearchRun(plan.search.run_id, 1e-4, 0.3),
                LrSearchRun(plan.search.run_id, 3e-4, 0.2),
                LrSearchRun(plan.search.run_id, 1e-3, 0.4),
                LrSearchRun(plan.search.run_id, 3e-3, 0.5),
                LrSearchRun(plan.search.run_id, 1e-2, 0.9),
                LrSearchRun(plan.search.run_id, 3e-5, 0.6),
            ),
        ),
    )
    name = checkpoint_name(protocol, 0, CONVNEXT)
    store.checkpoint_path(name).write_bytes(b"x")
    store.manifest_path(name).write_text(
        json.dumps(
            {"config": {"learning_rate": 3e-3, "valid_loss_history": [1.0], "best_epoch": 0}}
        )
    )

    report = status_report(store, [plan])

    (row,) = report.runs
    assert row["at_other_rate"] is True
    assert row["name"] == name
    assert row["chosen_lr"] == 3e-4
    assert "discard" in row["note"]
    assert "At another rate than the chosen one" in report.summary


def run_ensemble(monkeypatch, tmp_path, outcomes, seeds=(0, 1)):
    """Run the resumable ensemble with searches and training replaced by stand-ins."""
    searched, trained = [], []

    def fake_search(dls, search_protocol, builder, store, search_seed, accept, break_locks):
        searched.append(search_protocol.run_id)
        outcome = outcomes[search_protocol.run_id]
        if outcome is None:
            raise SearchBusy("candidates [0.003] are being run by another session")
        return search_won_by(outcome), False

    def fake_train(dls, winning, lr_search, seed, builder, store, checkpoint_suffix, break_locks):
        trained.append((checkpoint_suffix, seed, winning.learning_rate, winning.run_id))
        return "trained"

    monkeypatch.setattr(ocm_module, "load_or_search_learning_rate", fake_search)
    monkeypatch.setattr(ocm_module, "train_if_missing", fake_train)
    protocol = ocm_shared_protocol(1e-3)
    status = ocm_module.run_ocm_ensemble_resumable(
        None, protocol, seeds, make_store(tmp_path), progress=lambda _: None
    )
    return protocol, searched, trained, status


def test_each_backbone_is_searched_and_trained_at_its_own_rate(monkeypatch, tmp_path):
    protocol = ocm_shared_protocol(1e-3)
    outcomes = {
        ocm_search_protocol(protocol, REGNETY).run_id: 3e-3,
        ocm_search_protocol(protocol, CONVNEXT).run_id: 3e-4,
    }

    _, searched, trained, status = run_ensemble(monkeypatch, tmp_path, outcomes)

    assert len(searched) == 2 and len(set(searched)) == 2
    assert sorted(trained) == sorted(
        (backbone, seed, rate, protocol.run_id)
        for seed in (0, 1)
        for backbone, rate in ((REGNETY, 3e-3), (CONVNEXT, 3e-4))
    )
    assert set(status.values()) == {"trained"}


def test_a_backbone_whose_search_another_session_runs_waits_and_the_other_goes_on(
    monkeypatch, tmp_path
):
    protocol = ocm_shared_protocol(1e-3)
    outcomes = {
        ocm_search_protocol(protocol, REGNETY).run_id: 3e-3,
        ocm_search_protocol(protocol, CONVNEXT).run_id: None,
    }

    _, _, trained, status = run_ensemble(monkeypatch, tmp_path, outcomes)

    assert {backbone for backbone, *_ in trained} == {REGNETY}
    assert status[checkpoint_name(protocol, 0, CONVNEXT)] == "waiting for its learning-rate search"
    assert status[checkpoint_name(protocol, 1, REGNETY)] == "trained"
