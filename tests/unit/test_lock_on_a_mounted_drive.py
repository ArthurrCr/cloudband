"""Locks on a mounted Drive, which can show a file as it was before it was replaced.

On the real Drive a session took four runs in a row, each time reading back the lock it
had just written, got the old copy of the file, took it for another session's, reported
the run "busy" and moved on, leaving its own locks behind. Only the one run that had
no lock file yet was trained.
"""

import json
import time

import pytest

from cloudband.train.store import RunStore

NAMES = ["run-a", "run-b", "run-c", "run-d"]


def make_store(tmp_path):
    return RunStore(tmp_path / "results", tmp_path / "checkpoints", tmp_path / "models")


def other_session(store):
    return RunStore(store.results_dir, store.checkpoint_dir, store.source_dir)


def no_wait(_seconds):
    return None


def leave_old_lock(store, name, owner="deadbeef", age_seconds=7200):
    old = time.time() - age_seconds
    store.lock_path(name).write_text(
        json.dumps({"owner": owner, "started": old, "heartbeat": old})
    )
    return json.loads(store.lock_path(name).read_text())


def stale_view(monkeypatch, old_record, reads):
    """Make the first `reads` reads after the claim's own look return `old_record`."""
    real = RunStore.lock_info
    calls = {"n": 0}

    def lock_info(self, name):
        calls["n"] += 1
        if calls["n"] == 1:                  # the look claim takes before writing
            return real(self, name)
        if calls["n"] <= 1 + reads:
            return old_record
        return real(self, name)

    monkeypatch.setattr(RunStore, "lock_info", lock_info)


@pytest.mark.parametrize("reads", [1, 3, 7])
def test_an_old_copy_of_the_lock_does_not_make_the_run_busy(tmp_path, monkeypatch, reads):
    store = make_store(tmp_path)
    old = leave_old_lock(store, "run-a")
    stale_view(monkeypatch, old, reads)

    assert store.claim("run-a", sleep=no_wait) is True


def test_a_drive_that_never_shows_the_new_file_is_still_not_a_rival(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    old = leave_old_lock(store, "run-a")
    stale_view(monkeypatch, old, reads=1000)

    assert store.claim("run-a", sleep=no_wait) is True


def test_an_unreadable_lock_is_read_again(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    leave_old_lock(store, "run-a")
    stale_view(monkeypatch, None, reads=3)

    assert store.claim("run-a", sleep=no_wait) is True


def test_four_runs_with_old_locks_are_all_taken_not_skipped_as_busy(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    for name in NAMES:
        leave_old_lock(store, name)
    real = RunStore.lock_info

    def always_old(self, name):
        info = real(self, name)
        if info is not None and info["owner"] == self.session_id:
            return {"owner": "deadbeef", "started": 0, "heartbeat": 0}
        return info

    monkeypatch.setattr(RunStore, "lock_info", always_old)

    assert [store.claim(name, sleep=no_wait) for name in NAMES] == [True] * 4


def test_a_rival_that_wrote_after_us_is_still_a_rival(tmp_path):
    first = make_store(tmp_path)
    second = other_session(first)
    leave_old_lock(first, "run-a")

    def rival_writes_meanwhile(_seconds):
        second._write_lock("run-a")

    assert first.claim("run-a", sleep=rival_writes_meanwhile) is False
    assert "run-a" not in first._held


def test_a_live_lock_of_another_session_still_blocks_the_claim(tmp_path):
    first = make_store(tmp_path)
    second = other_session(first)
    first.claim("run-a", sleep=no_wait)

    assert second.claim("run-a", sleep=no_wait) is False


def test_the_heartbeat_does_not_need_to_read_the_lock_back(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    store.claim("run-a", sleep=no_wait)
    first = json.loads(store.lock_path("run-a").read_text())
    time.sleep(0.01)
    monkeypatch.setattr(RunStore, "lock_info", lambda self, name: None)

    store.heartbeat("run-a")

    renewed = json.loads(store.lock_path("run-a").read_text())
    assert renewed["heartbeat"] > first["heartbeat"]
    assert renewed["started"] == first["started"]


def test_the_heartbeat_does_not_overwrite_a_run_another_session_took(tmp_path):
    first = make_store(tmp_path)
    second = other_session(first)
    first.claim("run-a", sleep=no_wait)
    time.sleep(0.01)
    second.claim("run-a", break_locks=True, sleep=no_wait)

    first.heartbeat("run-a")

    assert json.loads(first.lock_path("run-a").read_text())["owner"] == second.session_id


def test_release_removes_the_lock_even_when_it_cannot_be_read_back(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    store.claim("run-a", sleep=no_wait)
    monkeypatch.setattr(RunStore, "lock_info", lambda self, name: None)

    store.release("run-a")

    assert not store.lock_path("run-a").exists()


def test_release_leaves_a_run_another_session_took_over(tmp_path):
    first = make_store(tmp_path)
    second = other_session(first)
    first.claim("run-a", sleep=no_wait)
    time.sleep(0.01)
    second.claim("run-a", break_locks=True, sleep=no_wait)

    first.release("run-a")

    assert json.loads(first.lock_path("run-a").read_text())["owner"] == second.session_id


def test_a_run_never_claimed_is_not_released_by_this_session(tmp_path):
    first = make_store(tmp_path)
    second = other_session(first)
    first.claim("run-a", sleep=no_wait)

    second.release("run-a")

    assert first.lock_path("run-a").exists()
