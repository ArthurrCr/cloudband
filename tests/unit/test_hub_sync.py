import os
import shutil
from pathlib import Path

import pytest

pytest.importorskip("huggingface_hub")

from cloudband.molab.hub_sync import HubMirror, local_files


class FakeHub:
    """A dataset repository kept in a folder, with the calls HubMirror makes."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.commits = []

    def create_repo(self, repo_id, repo_type, private, exist_ok):
        assert private and exist_ok and repo_type == "dataset"
        self.folder.mkdir(parents=True, exist_ok=True)

    def snapshot_download(self, repo_id, repo_type, local_dir):
        shutil.copytree(self.folder, local_dir, dirs_exist_ok=True)
        cache = Path(local_dir) / ".cache" / "huggingface"
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "metadata").write_text("x")

    def create_commit(self, repo_id, repo_type, operations, commit_message):
        self.commits.append(operations)
        for op in operations:
            target = self.folder / op.path_in_repo
            if type(op).__name__ == "CommitOperationAdd":
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(op.path_or_fileobj, target)
            else:
                target.unlink()


def make(tmp_path, now=1e10):
    hub = FakeHub(tmp_path / "hub")
    mirror = HubMirror("me/runs", tmp_path / "local", api=hub, clock=lambda: now)
    return hub, mirror


def write(path: Path, text: str, mtime: float = 1000.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    os.utime(path, (mtime, mtime))


def test_a_push_before_a_restore_is_refused(tmp_path):
    _, mirror = make(tmp_path)
    with pytest.raises(RuntimeError):
        mirror.push()


def test_restore_brings_files_down_and_a_push_right_after_sends_nothing(tmp_path):
    hub, mirror = make(tmp_path)
    write(hub.folder / "results" / "a.json", "{}")
    hub.folder.mkdir(exist_ok=True)

    assert mirror.restore() == 1
    report = mirror.push()

    assert not report.changed and hub.commits == []
    assert set(local_files(mirror.local_dir)) == {"results/a.json"}   # .cache skipped


def test_only_changed_files_are_sent_and_partials_never(tmp_path):
    hub, mirror = make(tmp_path)
    write(hub.folder / "checkpoints" / "old.pth", "old")
    mirror.restore()
    write(mirror.local_dir / "checkpoints" / "new.pth", "new")
    write(mirror.local_dir / "progress" / "x.state.pt.partial", "half")

    report = mirror.push()

    assert report.uploaded == ("checkpoints/new.pth",)
    assert (hub.folder / "checkpoints" / "new.pth").read_text() == "new"
    assert not (hub.folder / "progress").exists()
    assert not mirror.push().changed


def test_a_file_still_being_written_waits_until_it_settles(tmp_path):
    hub, mirror = make(tmp_path, now=1000.0 + 10)
    mirror.restore()
    write(mirror.local_dir / "checkpoints" / "busy.pth", "x", mtime=1000.0)

    assert mirror.push().waiting == ("checkpoints/busy.pth",)
    assert mirror.push_all().uploaded == ("checkpoints/busy.pth",)


def test_a_file_removed_locally_is_removed_from_the_hub(tmp_path):
    hub, mirror = make(tmp_path)
    write(hub.folder / "checkpoints" / "discarded.pth", "x")
    mirror.restore()
    (mirror.local_dir / "checkpoints" / "discarded.pth").unlink()

    report = mirror.push()

    assert report.deleted == ("checkpoints/discarded.pth",)
    assert not (hub.folder / "checkpoints" / "discarded.pth").exists()
