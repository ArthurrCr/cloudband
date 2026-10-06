import json
import time

import pytest
import torch

from cloudband.train.protocol import LrSearchRun, TrainProtocol
from cloudband.train.store import RunStore, ensure_shared_folder

PROTOCOL = TrainProtocol(run_id="ocm-rgn-cs12-shared", architecture="ocm",
                         learning_rate=1e-3)


def make_store(tmp_path):
    return RunStore(tmp_path / "results", tmp_path / "checkpoints", tmp_path / "models")


def other_session(store):
    """Another session on the same shared folder: same paths, another identity."""
    return RunStore(store.results_dir, store.checkpoint_dir, store.source_dir)


def no_wait(_seconds):
    return None


# ---- the state saved after every epoch ----------------------------------------


def test_a_state_round_trips_with_tensors_and_random_generator_state(tmp_path):
    store = make_store(tmp_path)
    state = {"model": {"w": torch.arange(4.0)}, "rng": {"python": (3, (1, 2, 3), None)},
             "valid_losses": [0.9, 0.5], "epochs_done": 2}

    store.save_epoch_state("run-a", state)
    loaded = store.load_epoch_state("run-a")

    assert torch.equal(loaded["model"]["w"], torch.arange(4.0))
    assert loaded["rng"] == {"python": (3, (1, 2, 3), None)}
    assert loaded["valid_losses"] == [0.9, 0.5]


def test_there_is_no_state_for_a_run_that_never_started(tmp_path):
    assert make_store(tmp_path).load_epoch_state("never") is None


def test_a_new_state_replaces_the_old_one_whole(tmp_path):
    store = make_store(tmp_path)
    store.save_epoch_state("run-a", {"epochs_done": 1})
    store.save_epoch_state("run-a", {"epochs_done": 2})

    assert store.load_epoch_state("run-a")["epochs_done"] == 2
    assert not list(store.progress_dir.glob("*.partial"))


def test_clearing_removes_the_state(tmp_path):
    store = make_store(tmp_path)
    store.save_epoch_state("run-a", {"epochs_done": 1})

    store.clear_epoch_state("run-a")
    store.clear_epoch_state("run-a")             # clearing twice is harmless

    assert store.load_epoch_state("run-a") is None


# ---- the search, candidate by candidate -----------------------------------------


def test_a_finished_candidate_is_kept_and_found_again(tmp_path):
    store = make_store(tmp_path)
    run = LrSearchRun("ocm-rgn-cs12-shared", 1e-3, 0.42)

    store.save_search_run(PROTOCOL, run, budget=(5, 5), seed=0)
    found = store.load_search_run(PROTOCOL, 1e-3, (5, 5), 0)

    assert found.val_loss == 0.42
    assert store.load_search_run(PROTOCOL, 3e-3, (5, 5), 0) is None


def test_candidates_accumulate_in_one_file(tmp_path):
    store = make_store(tmp_path)
    for rate, loss in ((1e-4, 0.5), (1e-3, 0.4), (3e-3, 0.45)):
        store.save_search_run(PROTOCOL, LrSearchRun("r", rate, loss), (5, 5), 0)

    records = json.loads(store.search_progress_path(PROTOCOL).read_text())

    assert [r["learning_rate"] for r in records] == [1e-4, 1e-3, 3e-3]


def test_a_result_made_with_another_budget_or_seed_is_refused(tmp_path):
    store = make_store(tmp_path)
    store.save_search_run(PROTOCOL, LrSearchRun("r", 1e-3, 0.4), (5, 5), 0)

    with pytest.raises(RuntimeError, match="budget"):
        store.load_search_run(PROTOCOL, 1e-3, (15, 15), 0)
    with pytest.raises(RuntimeError, match="seed"):
        store.load_search_run(PROTOCOL, 1e-3, (5, 5), 1)


# ---- one session at a time on a run ---------------------------------------------


def test_a_free_run_can_be_claimed(tmp_path):
    store = make_store(tmp_path)

    assert store.claim("run-a", sleep=no_wait) is True
    assert store.lock_info("run-a")["owner"] == store.session_id


def test_a_run_held_by_a_live_session_cannot_be_claimed(tmp_path):
    first = make_store(tmp_path)
    second = other_session(first)
    first.claim("run-a", sleep=no_wait)

    assert second.claim("run-a", sleep=no_wait) is False
    assert "last heartbeat" in second.describe_lock("run-a")


def test_a_session_can_claim_its_own_run_again(tmp_path):
    store = make_store(tmp_path)
    store.claim("run-a", sleep=no_wait)

    assert store.claim("run-a", sleep=no_wait) is True


def test_a_lock_nobody_renewed_is_taken_over(tmp_path):
    first = make_store(tmp_path)
    second = other_session(first)
    first.claim("run-a", sleep=no_wait)
    info = first.lock_info("run-a")
    info["heartbeat"] = time.time() - 7200
    first.lock_path("run-a").write_text(json.dumps(info))

    assert second.claim("run-a", lease_seconds=3600, sleep=no_wait) is True
    assert second.lock_info("run-a")["owner"] == second.session_id


def test_breaking_the_lock_takes_the_run_from_a_live_session(tmp_path):
    first = make_store(tmp_path)
    second = other_session(first)
    first.claim("run-a", sleep=no_wait)

    assert second.claim("run-a", break_locks=True, sleep=no_wait) is True


def test_a_heartbeat_renews_only_the_owners_lock(tmp_path):
    first = make_store(tmp_path)
    second = other_session(first)
    first.claim("run-a", sleep=no_wait)
    before = first.lock_info("run-a")["heartbeat"]
    time.sleep(0.01)

    second.heartbeat("run-a")                    # not the owner: no effect
    assert first.lock_info("run-a")["heartbeat"] == before
    first.heartbeat("run-a")
    assert first.lock_info("run-a")["heartbeat"] > before


def test_release_removes_only_the_owners_lock(tmp_path):
    first = make_store(tmp_path)
    second = other_session(first)
    first.claim("run-a", sleep=no_wait)

    second.release("run-a")
    assert first.lock_info("run-a") is not None
    first.release("run-a")
    assert first.lock_info("run-a") is None


def test_losing_the_race_to_another_session_is_noticed(tmp_path):
    first = make_store(tmp_path)
    second = other_session(first)

    def rival_writes_meanwhile(_seconds):
        second._write_lock("run-a")              # the other session wrote after us

    assert first.claim("run-a", sleep=rival_writes_meanwhile) is False


# ---- choices made once ------------------------------------------------------------


def test_a_setting_is_saved_for_every_session(tmp_path):
    first = make_store(tmp_path)
    first.save_setting("micro_batch", {"ocm": 4, "swin": 8})

    assert other_session(first).load_setting("micro_batch") == {"ocm": 4, "swin": 8}
    assert other_session(first).load_setting("absent", "default") == "default"


# ---- the shared folder --------------------------------------------------------------


def test_a_missing_shared_folder_says_how_to_create_the_shortcut(tmp_path):
    with pytest.raises(FileNotFoundError, match="Add shortcut to Drive"):
        ensure_shared_folder(tmp_path / "not-there")


def test_a_usable_shared_folder_is_returned(tmp_path):
    assert ensure_shared_folder(tmp_path) == tmp_path
    assert not list(tmp_path.glob(".write-test-*"))


def test_a_folder_that_cannot_be_written_to_is_reported(tmp_path, monkeypatch):
    def refuse(self, *args, **kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr(type(tmp_path), "write_text", refuse)

    with pytest.raises(PermissionError, match="editor, not as viewer"):
        ensure_shared_folder(tmp_path)