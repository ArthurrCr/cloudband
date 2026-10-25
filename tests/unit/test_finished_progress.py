"""Progress files of runs and candidates that are over, and the curves kept from them."""

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from cloudband.colab import drive_tidy
from cloudband.colab.drive_tidy import ROOT, SHARED, Copy, plan_tidy
from cloudband.train.loop import checkpoint_name
from cloudband.train.lr_search import proxy_protocol
from cloudband.train.protocol import LR_SEARCH_GRID, LrSearchRun, ocm_shared_protocol
from cloudband.train.status import Plan, finished_names
from cloudband.train.store import RunStore

BACKBONE = "regnety_004"
NOW = datetime(2026, 10, 19, 12, 0, tzinfo=timezone.utc)


def make_store(tmp_path):
    return RunStore(tmp_path / "results", tmp_path / "ckpt", tmp_path / "models")


def plan_for():
    protocol = replace(ocm_shared_protocol(1e-3), seeds=(0, 1))
    return Plan(f"OCM {BACKBONE}", protocol, BACKBONE)


def record(store, plan, rate, loss=0.4):
    store.save_search_run(
        plan.search, LrSearchRun(plan.search.run_id, rate, loss), (5, 5), plan.search.seeds[0]
    )


def candidate_name(plan, rate):
    return checkpoint_name(
        proxy_protocol(plan.search, rate), plan.search.seeds[0], f"lrsearch-{rate:g}"
    )


def test_curves_are_saved_and_read_back(tmp_path):
    store = make_store(tmp_path)
    plan = plan_for()

    store.save_curve(plan.search, 3e-4, [1.0, 0.8, 0.6])
    store.save_curve(plan.search, 1e-3, [0.9, 0.7])

    assert store.load_curves(plan.search) == {3e-4: [1.0, 0.8, 0.6], 1e-3: [0.9, 0.7]}


def test_a_finished_run_and_a_finished_candidate_with_its_curve_are_over(tmp_path):
    store = make_store(tmp_path)
    plan = plan_for()
    done = checkpoint_name(plan.protocol, 0, BACKBONE)
    store.checkpoint_path(done).write_bytes(b"x")
    store.manifest_path(done).write_text(json.dumps({"config": {}}))
    record(store, plan, LR_SEARCH_GRID[0])
    store.save_curve(plan.search, LR_SEARCH_GRID[0], [1.0] * 10)

    names = finished_names(store, [plan])

    assert done in names
    assert candidate_name(plan, LR_SEARCH_GRID[0]) in names
    assert checkpoint_name(plan.protocol, 1, BACKBONE) not in names


def test_a_candidate_whose_curve_is_not_saved_is_not_over_yet(tmp_path):
    store = make_store(tmp_path)
    plan = plan_for()
    record(store, plan, LR_SEARCH_GRID[0])
    record(store, plan, LR_SEARCH_GRID[1], loss=float("inf"))   # diverged: no curve to keep

    names = finished_names(store, [plan])

    assert candidate_name(plan, LR_SEARCH_GRID[0]) not in names
    assert candidate_name(plan, LR_SEARCH_GRID[1]) in names


def copy(id, name, size=500, modified="2026-10-10T00:00:00Z", mine=True, where=SHARED):
    return Copy(id, name, size, modified, mine, where)


def test_every_copy_of_the_progress_of_a_finished_run_goes_and_the_checkpoint_stays():
    copies = [
        copy("s1", "run-seed0.state.pt"),
        copy("s2", "run-seed0.state.pt", modified="2026-10-12T00:00:00Z"),
        copy("l1", "run-seed0.lock", size=85),
        copy("c1", "run-seed0.pth"),
        copy("t1", "run-seed1.state.pt", mine=False),
    ]

    plan = plan_tidy(copies, finished={"run-seed0"}, now=NOW)

    assert sorted(c.id for c in plan.trash) == ["l1", "s1", "s2"]
    assert [c.id for c in plan.others] == []
    assert sorted(c.id for c in plan.keep) == ["c1", "t1"]
    assert sorted(c.id for c in plan.of_finished) == ["l1", "s1", "s2"]


def test_progress_of_a_run_that_is_not_over_keeps_its_newest_copy():
    copies = [copy("a", "run-seed1.state.pt"),
              copy("b", "run-seed1.state.pt", modified="2026-10-12T00:00:00Z")]

    plan = plan_tidy(copies, finished={"run-seed0"}, now=NOW)

    assert [c.id for c in plan.keep] == ["b"]
    assert [c.id for c in plan.trash] == ["a"]


def test_an_interrupted_copy_goes_only_when_it_is_old():
    old = (NOW - timedelta(hours=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    fresh = (NOW - timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    copies = [copy("old", "a.state.pt.partial", modified=old, where=ROOT),
              copy("new", "b.state.pt.partial", modified=fresh)]

    plan = plan_tidy(copies, now=NOW)

    assert [c.id for c in plan.trash] == ["old"]
    assert [c.id for c in plan.keep] == ["new"]


def test_salvage_saves_the_curve_from_the_newest_complete_copy(tmp_path):
    store = make_store(tmp_path)
    plan = plan_for()
    record(store, plan, LR_SEARCH_GRID[0])
    record(store, plan, LR_SEARCH_GRID[1])
    state = f"{candidate_name(plan, LR_SEARCH_GRID[0])}.state.pt"
    copies = [copy("old", state, modified="2026-10-10T00:00:00Z"),
              copy("last", state, modified="2026-10-12T00:00:00Z")]
    curves = {"old": [1.0] * 6, "last": [1.0 - i / 20 for i in range(10)]}

    saved, missing = drive_tidy.salvage_curves(
        None, store, [plan], copies, read=lambda service, file_id: curves[file_id]
    )

    assert store.load_curves(plan.search)[LR_SEARCH_GRID[0]] == curves["last"]
    assert len(saved) == 1
    assert len(missing) == 1 and "no progress file left" in missing[0]


def test_salvage_refuses_a_curve_that_is_not_whole(tmp_path):
    store = make_store(tmp_path)
    plan = plan_for()
    record(store, plan, LR_SEARCH_GRID[0])
    state = f"{candidate_name(plan, LR_SEARCH_GRID[0])}.state.pt"

    saved, missing = drive_tidy.salvage_curves(
        None, store, [plan], [copy("only", state)], read=lambda service, file_id: [1.0] * 6
    )

    assert saved == []
    assert "6 of 10 epochs" in missing[0]
    assert store.load_curves(plan.search) == {}


def test_salvage_saves_a_short_curve_as_partial_only_when_accepted(tmp_path):
    store = make_store(tmp_path)
    plan = plan_for()
    record(store, plan, LR_SEARCH_GRID[0])
    state = f"{candidate_name(plan, LR_SEARCH_GRID[0])}.state.pt"
    copies = [copy("only", state)]

    saved, _ = drive_tidy.salvage_curves(
        None, store, [plan], copies, read=lambda service, file_id: [1.0] * 9, accept_partial=True
    )

    assert store.load_curves(plan.search)[LR_SEARCH_GRID[0]] == [1.0] * 9
    assert saved == [f"{plan.label} lr {LR_SEARCH_GRID[0]:g} (partial: 9 of 10 epochs)"]


def test_a_longer_curve_replaces_a_partial_one_and_a_shorter_one_does_not(tmp_path):
    store = make_store(tmp_path)
    plan = plan_for()
    record(store, plan, LR_SEARCH_GRID[0])
    store.save_curve(plan.search, LR_SEARCH_GRID[0], [1.0] * 8)
    state = f"{candidate_name(plan, LR_SEARCH_GRID[0])}.state.pt"
    copies = [copy("only", state)]

    drive_tidy.salvage_curves(
        None, store, [plan], copies, read=lambda service, file_id: [2.0] * 6, accept_partial=True
    )
    assert store.load_curves(plan.search)[LR_SEARCH_GRID[0]] == [1.0] * 8

    drive_tidy.salvage_curves(
        None, store, [plan], copies, read=lambda service, file_id: [3.0] * 10
    )
    assert store.load_curves(plan.search)[LR_SEARCH_GRID[0]] == [3.0] * 10


def test_a_partial_curve_frees_the_progress_files_only_when_accepted(tmp_path):
    store = make_store(tmp_path)
    plan = plan_for()
    record(store, plan, LR_SEARCH_GRID[0])
    store.save_curve(plan.search, LR_SEARCH_GRID[0], [1.0] * 9)
    name = candidate_name(plan, LR_SEARCH_GRID[0])

    assert name not in finished_names(store, [plan])
    assert name in finished_names(store, [plan], accept_partial=True)


def test_the_status_marks_a_partial_curve(tmp_path):
    from cloudband.train.status import search_rows

    store = make_store(tmp_path)
    plan = plan_for()
    record(store, plan, LR_SEARCH_GRID[0])
    record(store, plan, LR_SEARCH_GRID[1])
    store.save_curve(plan.search, LR_SEARCH_GRID[0], [1.0] * 9)
    store.save_curve(plan.search, LR_SEARCH_GRID[1], [1.0] * 10)

    rows = {r["lr"]: r["note"] for r in search_rows(store, plan.search, plan.label)}

    assert rows[LR_SEARCH_GRID[0]] == "curve partial: 9 of 10 epochs"
    assert rows[LR_SEARCH_GRID[1]] == ""


def test_the_tidy_asks_for_nothing_when_it_is_off(tmp_path, monkeypatch):
    def no_sign_in():
        raise AssertionError("the sign-in must not be asked for")

    monkeypatch.setattr(drive_tidy, "drive_service", no_sign_in)
    lines = []

    result = drive_tidy.run_tidy("off", make_store(tmp_path), [plan_for()], (), ("run",),
                                 say=lines.append)

    assert result is None
    assert "no Google sign-in" in lines[0]


def run_with_fakes(tmp_path, monkeypatch, mode):
    trashed = []
    mine = Copy("m", "x.lock", 85, "2026-10-10T00:00:00Z", True, SHARED)
    newest = Copy("n", "x.lock", 85, "2026-10-12T00:00:00Z", True, SHARED)
    monkeypatch.setattr(drive_tidy, "resolve_folder", lambda service, parts: "/".join(parts))
    monkeypatch.setattr(drive_tidy, "list_copies", lambda service, folders, prefixes: [mine, newest])
    monkeypatch.setattr(drive_tidy, "salvage_curves", lambda *a, **k: ([], []))
    monkeypatch.setattr(drive_tidy, "trash", lambda service, copies: trashed.extend(copies) or len(copies))
    lines = []
    plan = drive_tidy.run_tidy(mode, make_store(tmp_path), [plan_for()],
                               (tmp_path / "results",), ("x",), service=object(), say=lines.append)
    return plan, trashed, lines


def test_list_mode_shows_what_would_go_and_trashes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(drive_tidy, "folder_parts", lambda path: ("results",))

    plan, trashed, lines = run_with_fakes(tmp_path, monkeypatch, "list")

    assert [c.id for c in plan.trash] == ["m"]
    assert trashed == []
    assert any("nothing was trashed" in line for line in lines)


def test_trash_mode_trashes_the_superseded_copies(tmp_path, monkeypatch):
    monkeypatch.setattr(drive_tidy, "folder_parts", lambda path: ("results",))

    _, trashed, _ = run_with_fakes(tmp_path, monkeypatch, "trash")

    assert [c.id for c in trashed] == ["m"]


def test_an_unknown_mode_is_refused(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        drive_tidy.run_tidy("yes", make_store(tmp_path), [], (), ())
