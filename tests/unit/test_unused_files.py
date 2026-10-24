import json
import os
import time

from cloudband.train.store import RunStore

HOUR = 3600.0


def make_store(tmp_path):
    return RunStore(tmp_path / "results", tmp_path / "ckpt", tmp_path / "models", session_id="me")


def age(path, seconds):
    old = time.time() - seconds
    os.utime(path, (old, old))
    return path


def lock(store, name, heartbeat_ago):
    store.lock_path(name).write_text(
        json.dumps({"owner": "x", "started": 0, "heartbeat": time.time() - heartbeat_ago})
    )
    age(store.lock_path(name), heartbeat_ago)


def finish(store, name):
    store.checkpoint_path(name).write_bytes(b"x")
    store.manifest_path(name).write_text("{}")
    age(store.checkpoint_path(name), 5 * HOUR)
    age(store.manifest_path(name), 5 * HOUR)


def listed(store):
    return {p.name: reason for p, reason in store.unused_files()}


def test_leftovers_of_a_finished_run_are_listed(tmp_path):
    store = make_store(tmp_path)
    finish(store, "done-run")
    state = store.epoch_state_path("done-run")
    state.write_bytes(b"s")
    age(state, 5 * HOUR)
    lock(store, "done-run", 5 * HOUR)

    assert set(listed(store)) == {"done-run.state.pt", "done-run.lock"}


def test_a_stale_lock_without_progress_is_listed_but_progress_is_kept(tmp_path):
    store = make_store(tmp_path)
    lock(store, "no-progress", 5 * HOUR)
    lock(store, "paused", 5 * HOUR)
    state = store.epoch_state_path("paused")
    state.write_bytes(b"s")
    age(state, 5 * HOUR)

    assert set(listed(store)) == {"no-progress.lock"}


def test_a_run_with_a_live_lock_is_never_listed(tmp_path):
    store = make_store(tmp_path)
    finish(store, "live")
    lock(store, "live", 60)
    state = store.epoch_state_path("live")
    state.write_bytes(b"s")
    age(state, 5 * HOUR)
    partial = store.checkpoint_path("live").with_suffix(".pth.partial")
    partial.write_bytes(b"p")
    age(partial, 5 * HOUR)

    assert "live.state.pt" not in listed(store)
    assert "live.lock" not in listed(store)


def test_recent_files_are_left_alone(tmp_path):
    store = make_store(tmp_path)
    (store.results_dir / "a.json.partial").write_text("p")   # just written

    assert listed(store) == {}


def test_halves_and_partials_are_listed_but_search_records_and_settings_are_not(tmp_path):
    store = make_store(tmp_path)
    for path in (
        store.checkpoint_path("orphan-ckpt"),
        store.manifest_path("orphan-manifest"),
        store.results_dir / "x-lr-search.json.partial",
        store.results_dir / "x-lr-search.json",
        store.results_dir / "x-lr-search.partial.json",
        store.results_dir / "settings.json",
        store.results_dir / ".write-test-abc",
    ):
        path.write_text("{}")
        age(path, 5 * HOUR)

    assert set(listed(store)) == {
        "orphan-ckpt.pth",
        "orphan-manifest.json",
        "x-lr-search.json.partial",
        ".write-test-abc",
    }


def test_saved_validation_curves_are_never_listed(tmp_path):
    store = make_store(tmp_path)
    for name in ("ocm-x-lr-curves.json", "ocm-x-convnextv2_nano.fcmae_ft_in1k-lr-curves.json"):
        path = store.results_dir / name
        path.write_text("{}")
        age(path, 5 * HOUR)

    assert listed(store) == {}


def test_remove_unused_deletes_only_what_was_listed(tmp_path):
    store = make_store(tmp_path)
    finish(store, "keep")
    lock(store, "gone", 5 * HOUR)

    removed = store.remove_unused(store.unused_files())

    assert removed == 1
    assert store.is_done("keep")
    assert not store.lock_path("gone").exists()


def test_an_entry_the_drive_cannot_read_is_skipped_and_reported(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    lock(store, "gone", 5 * HOUR)
    lock(store, "ghost", 5 * HOUR)
    ghost = store.lock_path("ghost")
    real_stat = type(ghost).stat

    def stat(self, *args, **kwargs):
        if self.name == "ghost.lock":
            raise OSError(107, "Transport endpoint is not connected")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(type(ghost), "stat", stat)
    skipped = []

    found = {p.name for p, _ in store.unused_files(skipped=skipped)}

    assert found == {"gone.lock"}
    assert [p.name for p in skipped] == ["ghost.lock"]


def test_the_status_report_survives_an_entry_the_drive_cannot_read(tmp_path, monkeypatch):
    from dataclasses import replace

    from cloudband.train.protocol import ocm_shared_protocol
    from cloudband.train.status import Plan, status_report

    store = make_store(tmp_path)
    protocol = replace(ocm_shared_protocol(1e-3), seeds=(0,))
    real_is_file = type(store.results_dir).is_file

    def is_file(self, *args, **kwargs):
        if self.suffix == ".json" and "seed0" in self.name:
            raise OSError(107, "Transport endpoint is not connected")
        return real_is_file(self, *args, **kwargs)

    monkeypatch.setattr(type(store.results_dir), "is_file", is_file)

    report = status_report(store, [Plan("OCM", protocol, "regnety_004")])

    assert report.runs[0]["state"] == "not started"
