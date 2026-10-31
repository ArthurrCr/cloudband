"""A mounted Drive that was idle fails its first access with "Transport endpoint is not
connected" (errno 107) and answers the next one. Seen in Colab right after the local
cache was built: the training cell failed on its first read and worked when run again.
"""

import errno
import json
import os
import pathlib
import time

import pytest

from cloudband.train import store as store_module
from cloudband.train.protocol import LrSearchManifest, LrSearchRun, ocm_shared_protocol
from cloudband.train.store import DriveDisconnected, RunStore


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    waits = []
    monkeypatch.setattr(store_module, "_sleep", waits.append)
    return waits


def make_store(tmp_path):
    return RunStore(tmp_path / "results", tmp_path / "checkpoints", tmp_path / "models")


def not_connected(path):
    return OSError(errno.ENOTCONN, "Transport endpoint is not connected", str(path))


def failing(monkeypatch, method, target, times):
    """Make Path.<method> fail with errno 107 on `target` for its first `times` calls."""
    original = getattr(pathlib.Path, method)
    calls = {"n": 0}

    def patched(self, *args, **kwargs):
        if self == target and calls["n"] < times:
            calls["n"] += 1
            raise not_connected(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, method, patched)
    return calls


def mount_down(monkeypatch):
    def refuse(_folder):
        raise not_connected(_folder)

    monkeypatch.setattr(os, "listdir", refuse)


def save_a_search(store):
    protocol = ocm_shared_protocol(learning_rate=1e-3)
    search = LrSearchManifest(
        architecture="ocm",
        runs=(LrSearchRun(run_id=protocol.run_id, learning_rate=1e-3, val_loss=0.3),),
    )
    store.save_search(protocol, search)
    return protocol


def test_an_idle_mount_is_read_again_instead_of_failing(tmp_path, monkeypatch, no_waiting):
    store = make_store(tmp_path)
    protocol = save_a_search(store)
    failing(monkeypatch, "is_file", store.search_path(protocol), times=1)

    search = store.load_search(protocol)

    assert search.winner().learning_rate == 1e-3
    assert no_waiting == [store_module.RETRY_WAITS[0]]


def test_a_read_that_fails_once_is_read_again(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    protocol = save_a_search(store)
    failing(monkeypatch, "read_text", store.search_path(protocol), times=2)

    assert store.load_search(protocol) is not None


def test_a_mount_that_stays_down_stops_with_how_to_remount(tmp_path, monkeypatch, no_waiting):
    store = make_store(tmp_path)
    protocol = save_a_search(store)
    failing(monkeypatch, "is_file", store.search_path(protocol), times=100)
    mount_down(monkeypatch)

    with pytest.raises(DriveDisconnected, match="force_remount=True"):
        store.load_search(protocol)
    assert no_waiting == list(store_module.RETRY_WAITS)


def test_an_entry_trashed_from_outside_counts_as_absent(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    protocol = save_a_search(store)
    failing(monkeypatch, "is_file", store.search_path(protocol), times=100)

    assert store.load_search(protocol) is None


def test_other_errors_are_not_retried(tmp_path, monkeypatch, no_waiting):
    store = make_store(tmp_path)
    protocol = save_a_search(store)
    target = store.search_path(protocol)
    original = pathlib.Path.read_text

    def denied(self, *args, **kwargs):
        if self == target:
            raise PermissionError(errno.EACCES, "denied", str(self))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "read_text", denied)

    with pytest.raises(PermissionError):
        store.load_search(protocol)
    assert no_waiting == []


def test_an_unreadable_lock_is_never_taken_for_a_free_run(tmp_path, monkeypatch):
    """Reading "no lock" from a dropped mount would let this session take a run that
    another session is training."""
    store = make_store(tmp_path)
    now = time.time()
    store.lock_path("run").write_text(
        json.dumps({"owner": "other", "started": now, "heartbeat": now})
    )
    failing(monkeypatch, "is_file", store.lock_path("run"), times=100)
    mount_down(monkeypatch)

    with pytest.raises(DriveDisconnected):
        store.claim("run", sleep=lambda _s: None)
    assert json.loads(store.lock_path("run").read_text())["owner"] == "other"


def test_a_lock_read_once_more_still_keeps_the_other_session_out(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    now = time.time()
    store.lock_path("run").write_text(
        json.dumps({"owner": "other", "started": now, "heartbeat": now})
    )
    failing(monkeypatch, "read_text", store.lock_path("run"), times=1)

    assert store.claim("run", sleep=lambda _s: None) is False


def test_a_finished_run_is_still_found_after_a_hiccup(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    store.manifest_path("run").write_text("{}")
    store.checkpoint_path("run").write_bytes(b"weights")
    failing(monkeypatch, "is_file", store.checkpoint_path("run"), times=1)

    assert store.is_done("run")
