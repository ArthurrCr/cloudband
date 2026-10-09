import json
import time
from dataclasses import replace

from cloudband.train.loop import checkpoint_name
from cloudband.train.lr_search import proxy_protocol
from cloudband.train.protocol import (
    LR_SEARCH_GRID,
    LrSearchManifest,
    LrSearchRun,
    ocm_shared_protocol,
    swin_shared_protocol,
)
from cloudband.train.status import LEASE_SECONDS, Plan, status_report
from cloudband.train.store import RunStore

BACKBONE = "regnety_004"


def make_store(tmp_path, session="aaaa"):
    return RunStore(
        results_dir=tmp_path / "results",
        checkpoint_dir=tmp_path / "checkpoints",
        source_dir=tmp_path / "models",
        session_id=session,
    )


def protocols():
    ocm = replace(ocm_shared_protocol(1e-3), seeds=(0, 1))
    swin = replace(swin_shared_protocol(1e-3), seeds=(0, 1))
    return ocm, swin


def plans():
    ocm, swin = protocols()
    return [Plan(f"OCM {BACKBONE}", ocm, BACKBONE), Plan("Swin", swin)]


def finish(store, name, history, lr=1e-3):
    store.checkpoint_path(name).write_bytes(b"x")
    best = min(range(len(history)), key=lambda i: history[i])
    store.manifest_path(name).write_text(
        json.dumps(
            {
                "config": {
                    "valid_loss_history": history,
                    "best_epoch": best,
                    "learning_rate": lr,
                }
            }
        )
    )


def in_flight(store, name, losses, lr=1e-3):
    store.save_epoch_state(
        name, {"valid_losses": losses, "fingerprint": {"learning_rate": lr}}
    )


def save_search(store, protocol, losses):
    store.save_search(
        protocol,
        LrSearchManifest(
            architecture=protocol.architecture,
            runs=tuple(
                LrSearchRun(run_id=protocol.run_id, learning_rate=lr, val_loss=loss)
                for lr, loss in zip(LR_SEARCH_GRID, losses, strict=True)
            ),
        ),
    )


def row(report, model, seed):
    return next(r for r in report.runs if r["model"] == model and r["seed"] == seed)


def test_an_empty_folder_reports_everything_as_not_started(tmp_path):
    report = status_report(make_store(tmp_path), plans())

    assert {r["state"] for r in report.runs} == {"not started"}
    assert {r["state"] for r in report.search} == {"not started"}
    assert len(report.runs) == 4
    assert "epochs: 0 of 120 (0%)" in report.summary
    assert "Running now: nothing" in report.summary


def test_a_finished_run_shows_its_epochs_best_loss_and_rate(tmp_path):
    store = make_store(tmp_path)
    ocm, _ = protocols()
    name = checkpoint_name(ocm, 0, BACKBONE)
    finish(store, name, [1.0, 0.8, 0.5, 0.6] + [0.9] * 26, lr=3e-4)

    result = row(status_report(store, plans()), f"OCM {BACKBONE}", 0)

    assert result["state"] == "done"
    assert result["epochs"] == "30/30"
    assert result["best_val_loss"] == 0.5
    assert result["lr"] == 3e-4
    assert result["note"] == ""


def test_a_run_whose_best_epoch_is_among_the_last_is_flagged(tmp_path):
    store = make_store(tmp_path)
    ocm, _ = protocols()
    finish(store, checkpoint_name(ocm, 0, BACKBONE), [1.0 - i * 0.01 for i in range(30)])

    report = status_report(store, plans())

    assert "still improving" in row(report, f"OCM {BACKBONE}", 0)["note"]
    assert "Still improving at the end: 1" in report.summary


def test_a_run_with_a_recent_lock_is_running_and_names_the_session(tmp_path):
    store = make_store(tmp_path, session="mine")
    _, swin = protocols()
    name = checkpoint_name(swin, 1)
    in_flight(store, name, [1.2, 1.1, 1.0])
    assert store.claim(name, sleep=lambda _: None)

    report = status_report(store, plans())
    result = row(report, "Swin", 1)

    assert result["state"] == "running"
    assert result["epochs"] == "3/30"
    assert result["best_val_loss"] == 1.0
    assert "mine (this session)" in result["session"]
    assert "Swin seed 1 (3/30" in report.summary


def test_a_run_nobody_renews_is_paused_and_says_where_it_continues(tmp_path):
    store = make_store(tmp_path)
    ocm, _ = protocols()
    name = checkpoint_name(ocm, 0, BACKBONE)
    in_flight(store, name, [1.2, 1.1])
    assert store.claim(name, sleep=lambda _: None)
    later = time.time() + LEASE_SECONDS + 600

    report = status_report(store, plans(), now=later)
    result = row(report, f"OCM {BACKBONE}", 0)

    assert result["state"] == "paused"
    assert result["note"] == "continues from epoch 2"
    assert "Paused, to continue" in report.summary


def test_a_search_in_two_pieces_shows_done_running_and_missing_candidates(tmp_path):
    store = make_store(tmp_path, session="mine")
    ocm, _ = protocols()
    for lr, loss in zip(LR_SEARCH_GRID[:2], (1.3, 1.1)):
        store.save_search_run(
            ocm, LrSearchRun(run_id=ocm.run_id, learning_rate=lr, val_loss=loss), (5, 5), 0
        )
    third = checkpoint_name(proxy_protocol(ocm, LR_SEARCH_GRID[2]), 0, f"lrsearch-{LR_SEARCH_GRID[2]:g}")
    in_flight(store, third, [1.4, 1.3, 1.2], lr=LR_SEARCH_GRID[2])
    assert store.claim(third, sleep=lambda _: None)

    rows = [r for r in status_report(store, plans()).search if r["model"] == "ocm"]

    assert [r["state"] for r in rows] == ["done", "done", "running", "not started", "not started"]
    assert rows[1]["best_val_loss"] == 1.1
    assert rows[2]["epochs"] == "3/10"


def test_a_finished_search_names_its_winner_and_warns_about_the_edge(tmp_path):
    store = make_store(tmp_path)
    ocm, swin = protocols()
    save_search(store, ocm, (1.5, 1.2, 1.0, 1.1, 1.3))
    save_search(store, swin, (0.9, 1.2, 1.3, 1.4, 1.5))

    report = status_report(store, plans())
    notes = {r["model"]: r["note"] for r in report.search if r["note"]}

    assert notes["ocm"] == "winner"
    assert notes["swin"] == "winner (edge of the grid)"
    assert "winner 0.0003" in report.summary
    # runs that have not started take the winning rate as the one they will use
    assert row(report, f"OCM {BACKBONE}", 0)["lr"] == LR_SEARCH_GRID[2]


def test_seeds_are_finished_only_when_every_model_has_them(tmp_path):
    store = make_store(tmp_path)
    ocm, swin = protocols()
    finish(store, checkpoint_name(ocm, 0, BACKBONE), [1.0] * 30)
    finish(store, checkpoint_name(ocm, 1, BACKBONE), [1.0] * 30)
    finish(store, checkpoint_name(swin, 0), [1.0] * 30)

    report = status_report(store, plans())

    assert "seeds finished for every model: [0]" in report.summary
    assert "epochs: 90 of 120 (75%)" in report.summary


def test_an_unreadable_progress_file_does_not_stop_the_report(tmp_path):
    store = make_store(tmp_path)
    ocm, _ = protocols()
    store.epoch_state_path(checkpoint_name(ocm, 0, BACKBONE)).write_bytes(b"not a state")

    result = row(status_report(store, plans()), f"OCM {BACKBONE}", 0)

    assert result["state"] == "paused"
    assert "unreadable" in result["note"]


def test_the_report_changes_nothing_in_the_folder(tmp_path):
    store = make_store(tmp_path)
    ocm, _ = protocols()
    in_flight(store, checkpoint_name(ocm, 0, BACKBONE), [1.0])
    before = sorted((p, p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file())

    status_report(store, plans())

    assert sorted((p, p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()) == before
