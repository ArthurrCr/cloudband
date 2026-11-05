import itertools
import os
import shutil
from pathlib import Path

import pytest

from cloudband.colab.drive_tidy import FOLDER, SHORTCUT
from cloudband.molab.drive_sync import DriveMirror, code_from, local_files, pick_copy


_ids = (f"new{i}" for i in itertools.count())
_times = (f"2030-01-01T00:00:{i:06d}Z" for i in itertools.count())


class FakeDrive(DriveMirror):
    """A DriveMirror whose Drive is a dict of entries, with no network."""

    def __init__(self, local_dir, entries, **kwargs):
        kwargs.setdefault("clock", lambda: 1e10)
        super().__init__(service=None, local_dir=local_dir, log=lambda _m: None, **kwargs)
        self.entries = entries          # id -> dict(name, mimeType, parent, data, ...)
        self.ids = _ids
        self.trashed = []

    def _children(self, folder_id):
        return [
            {"id": key, **{k: v for k, v in e.items() if k not in ("parent", "data")}}
            for key, e in self.entries.items()
            if e["parent"] == folder_id
        ]

    def _create_folder(self, name, parent_id):
        key = next(self.ids)
        self.entries[key] = {"name": name, "mimeType": FOLDER, "parent": parent_id}
        return key

    def _download(self, file_id, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.entries[file_id]["data"])

    def _upload(self, path, name, parent_id, file_id):
        key = file_id or next(self.ids)
        self.entries[key] = {
            "name": name, "mimeType": "application/octet-stream", "parent": parent_id,
            "data": path.read_bytes(), "size": str(path.stat().st_size),
            "modifiedTime": next(_times),
        }
        return {"id": key, "modifiedTime": self.entries[key]["modifiedTime"]}

    def _trash(self, file_id):
        if file_id in self.entries:
            self.trashed.append(self.entries.pop(file_id)["name"])


def file(name, parent, data, modified="2026-01-01T00:00:00Z"):
    return {"name": name, "mimeType": "application/octet-stream", "parent": parent,
            "data": data, "size": str(len(data)), "modifiedTime": modified}


def colab_drive():
    """My Drive with cloudband_treino as a shortcut to a shared folder, as on Colab."""
    return {
        "short": {"name": "cloudband_treino", "mimeType": SHORTCUT, "parent": "root",
                  "shortcutDetails": {"targetId": "shared", "targetMimeType": FOLDER}},
        "res": {"name": "results", "mimeType": FOLDER, "parent": "shared"},
        "chk": {"name": "checkpoints", "mimeType": FOLDER, "parent": "shared"},
        "prog": {"name": "progress", "mimeType": FOLDER, "parent": "res"},
        "other": {"name": "notes", "mimeType": FOLDER, "parent": "shared"},
        "s": file("settings.json", "res", b"{}"),
        "old": file("a.state.pt", "prog", b"x" * 100, "2026-01-01T00:00:00Z"),
        "new": file("a.state.pt", "prog", b"y" * 100, "2026-01-02T00:00:00Z"),
        "cut": file("a.state.pt", "prog", b"z" * 10, "2026-01-03T00:00:00Z"),
        "c": file("ocm.pth", "chk", b"weights"),
        "p": file("b.lock.partial", "prog", b"half"),
        "n": file("big.bin", "other", b"not mirrored"),
    }


@pytest.fixture
def no_folder_resolution(monkeypatch):
    import cloudband.molab.drive_sync as module

    def resolve(service, parts):
        assert parts == ("cloudband_treino",)
        return "shared"

    monkeypatch.setattr(module, "resolve_path", resolve)


def write(path: Path, text: str, mtime: float = 1000.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    os.utime(path, (mtime, mtime))


def test_the_newest_complete_copy_is_the_one_used():
    copies = [
        {"size": "100", "modifiedTime": "2026-01-01"},
        {"size": "100", "modifiedTime": "2026-01-02"},
        {"size": "10", "modifiedTime": "2026-01-03"},
    ]
    assert pick_copy(copies)["modifiedTime"] == "2026-01-02"


def test_restore_brings_down_results_and_checkpoints_only(tmp_path, no_folder_resolution):
    mirror = FakeDrive(tmp_path / "local", colab_drive(), clock=lambda: 1e10)

    assert mirror.restore() == 3
    assert set(local_files(mirror.local_dir)) == {
        "results/settings.json", "results/progress/a.state.pt", "checkpoints/ocm.pth",
    }
    assert (mirror.local_dir / "results/progress/a.state.pt").read_bytes() == b"y" * 100
    assert not (mirror.local_dir / "notes").exists()
    assert not mirror.push().changed


def test_a_push_before_a_restore_is_refused(tmp_path):
    with pytest.raises(RuntimeError):
        FakeDrive(tmp_path, {}).push()


def test_changes_go_up_in_place_and_new_files_in_their_folder(tmp_path, no_folder_resolution):
    drive = colab_drive()
    mirror = FakeDrive(tmp_path / "local", drive, clock=lambda: 1e10)
    mirror.restore()
    write(mirror.local_dir / "checkpoints/ocm.pth", "better weights")
    write(mirror.local_dir / "results/progress/b.lock", "{}")
    write(mirror.local_dir / "results/progress/c.state.pt.partial", "half")

    report = mirror.push()

    assert set(report.uploaded) == {"checkpoints/ocm.pth", "results/progress/b.lock"}
    assert drive["c"]["data"] == b"better weights"           # same file id, updated
    lock = next(e for e in drive.values() if e["name"] == "b.lock")
    assert lock["parent"] == "prog"
    assert not any(e["name"].startswith("c.state") for e in drive.values())


def test_a_new_subfolder_is_created_on_the_drive(tmp_path, no_folder_resolution):
    drive = colab_drive()
    mirror = FakeDrive(tmp_path / "local", drive, clock=lambda: 1e10)
    mirror.restore()
    write(mirror.local_dir / "results/extra/x.csv", "1")

    mirror.push()

    folder = next(k for k, e in drive.items() if e["name"] == "extra")
    assert drive[folder]["parent"] == "res"
    assert any(e["name"] == "x.csv" and e["parent"] == folder for e in drive.values())


def test_a_recent_file_waits_until_it_settles(tmp_path, no_folder_resolution):
    mirror = FakeDrive(tmp_path / "local", colab_drive(), clock=lambda: 1010.0)
    mirror.restore()
    write(mirror.local_dir / "checkpoints/busy.pth", "x", mtime=1000.0)

    assert mirror.push().waiting == ("checkpoints/busy.pth",)
    assert mirror.push_all().uploaded == ("checkpoints/busy.pth",)


def test_a_file_removed_locally_goes_to_the_drive_trash(tmp_path, no_folder_resolution):
    mirror = FakeDrive(tmp_path / "local", colab_drive(), clock=lambda: 1e10)
    mirror.restore()
    (mirror.local_dir / "checkpoints/ocm.pth").unlink()

    assert mirror.push().deleted == ("checkpoints/ocm.pth",)
    assert mirror.trashed == ["ocm.pth"]


def test_the_code_is_read_from_the_address_the_browser_was_sent_to():
    url = "http://localhost/?state=abc&code=4/0AbC-d_e&scope=https://www.googleapis.com/auth/drive"
    assert code_from(url) == "4/0AbC-d_e"
    assert code_from("  4/0raw  ") == "4/0raw"


def test_a_pull_brings_what_another_session_sent_and_drops_what_it_removed(
    tmp_path, no_folder_resolution
):
    drive = colab_drive()
    mine = FakeDrive(tmp_path / "mine", drive)
    other = FakeDrive(tmp_path / "other", drive)
    mine.restore()
    other.restore()
    write(other.local_dir / "results/progress/x.lock", "theirs")
    other.push_all()

    assert "results/progress/x.lock" in mine.pull()
    assert (mine.local_dir / "results/progress/x.lock").read_text() == "theirs"

    (other.local_dir / "results/progress/x.lock").unlink()
    other.push_all()
    mine.pull()
    assert not (mine.local_dir / "results/progress/x.lock").exists()


def test_a_change_made_here_and_not_yet_sent_is_not_overwritten_by_a_pull(
    tmp_path, no_folder_resolution
):
    drive = colab_drive()
    mine = FakeDrive(tmp_path / "mine", drive)
    other = FakeDrive(tmp_path / "other", drive)
    mine.restore()
    other.restore()
    write(other.local_dir / "results/settings.json", "theirs")
    other.push_all()
    write(mine.local_dir / "results/settings.json", "mine")

    mine.pull()

    assert (mine.local_dir / "results/settings.json").read_text() == "mine"


# ---- two sessions sharing the Drive through DriveRunStore ----

from cloudband.molab.drive_sync import DriveRunStore  # noqa: E402


def session(tmp_path, label, drive):
    mirror = FakeDrive(tmp_path / label, drive)
    mirror.restore()
    store = DriveRunStore(
        mirror.local_dir / "results", mirror.local_dir / "checkpoints",
        tmp_path / label / "models", mirror=mirror,
    )
    return store


def no_wait(_seconds):
    return None


def test_a_run_taken_by_one_session_is_refused_to_the_other(tmp_path, no_folder_resolution):
    drive = colab_drive()
    first, second = session(tmp_path, "a", drive), session(tmp_path, "b", drive)

    assert first.claim("run-seed0", sleep=no_wait)
    assert not second.claim("run-seed0", sleep=no_wait)

    first.release("run-seed0")
    assert second.claim("run-seed0", sleep=no_wait)


def test_a_run_finished_by_another_session_is_not_taken_again(tmp_path, no_folder_resolution):
    drive = colab_drive()
    first, second = session(tmp_path, "a", drive), session(tmp_path, "b", drive)
    assert first.claim("run-seed0", sleep=no_wait)
    write(first.checkpoint_path("run-seed0"), "weights")
    write(first.manifest_path("run-seed0"), "{}")
    first.mirror.push_all()
    first.release("run-seed0")

    assert not second.claim("run-seed0", sleep=no_wait)
    assert second.is_done("run-seed0")


def test_settings_written_by_two_sessions_are_both_kept(tmp_path, no_folder_resolution):
    drive = colab_drive()
    first, second = session(tmp_path, "a", drive), session(tmp_path, "b", drive)

    first.save_setting("micro_batch", {"ocm": 8})
    second.save_setting("other", 1)
    first.mirror.pull()

    assert first.load_setting("micro_batch") == {"ocm": 8}
    assert first.load_setting("other") == 1


def test_a_worker_starts_from_the_notebook_copy_without_downloading_again(
    tmp_path, no_folder_resolution
):
    drive = colab_drive()
    notebook = FakeDrive(tmp_path / "notebook", drive)
    notebook.restore()
    notebook.save_state()
    worker = FakeDrive(tmp_path / "worker", drive)
    downloads = []
    worker._download = lambda file_id, dest: downloads.append(file_id)

    assert worker.seed_from(notebook.local_dir) == 3
    assert downloads == []
    assert (worker.local_dir / "checkpoints/ocm.pth").read_bytes() == b"weights"

    write(worker.local_dir / "results/progress/w.lock", "worker")
    worker.push_all()
    assert not (notebook.local_dir / "results/progress/w.lock").exists()
    assert (notebook.local_dir / "checkpoints/ocm.pth").read_bytes() == b"weights"
class NotAllowed(Exception):
    class resp:
        status = 403


def test_a_lock_owned_by_another_account_is_released_by_overwriting_it(
    tmp_path, no_folder_resolution
):
    drive = colab_drive()
    drive["theirs"] = file("run.lock", "prog", b'{"owner": "colab"}')
    drive["state"] = file("run.state.pt", "prog", b"s" * 100)
    mirror = FakeDrive(tmp_path / "local", drive)

    def refuse(file_id):
        raise NotAllowed()

    mirror._trash = refuse
    mirror.restore()
    store = DriveRunStore(
        mirror.local_dir / "results", mirror.local_dir / "checkpoints",
        tmp_path / "models", mirror=mirror,
    )
    assert store.claim("run", sleep=no_wait, break_locks=True)
    store.release("run")
    (mirror.local_dir / "results/progress/run.state.pt").unlink()
    mirror.push_all()

    import json
    assert json.loads(drive["theirs"]["data"])["heartbeat"] == 0
    assert store.lock_info("run") is None                 # not brought back here
    assert "state" in drive                               # kept for its owner
    assert not (mirror.local_dir / "results/progress/run.state.pt").exists()
    mirror.pull()
    assert not (mirror.local_dir / "results/progress/run.state.pt").exists()

    other = session(tmp_path, "other", drive)
    assert other.claim("run", sleep=no_wait)              # free for everyone
