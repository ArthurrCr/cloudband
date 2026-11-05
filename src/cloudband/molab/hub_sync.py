"""Keep the training folder of a molab session on the Hugging Face Hub.

On Colab the runs live in a mounted Google Drive folder, so a session that ends
loses nothing. A molab container has no such mount: its disk goes away with the
session. RunStore still needs a plain folder, so the session works in a local
folder and this module mirrors it to a private dataset repository on the Hub.

The mirror is one-way at a time: `restore` brings the repository down once when
the session starts, and every `push` sends up what changed since. Only files
that changed (by size or modification time) are sent, so a push after an epoch
uploads the new state and checkpoint, not the whole folder again. A file still
being written (a `.partial`, or one modified in the last `settle_seconds`) waits
for the next push. A file removed locally, for instance by RunStore.discard, is
removed from the repository too.

The repository is the only copy that outlives the session, so a push refuses to
run before a successful restore: an empty local folder must never be taken as
"everything was deleted". Two sessions must not train on the same repository at
the same time, since each would overwrite the other's pushes.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

SKIPPED_DIRS = (".cache", ".git", ".ipynb_checkpoints")
SKIPPED_FILES = (".gitattributes",)
SKIPPED_SUFFIXES = (".partial",)


@dataclass(frozen=True)
class PushReport:
    uploaded: tuple[str, ...]
    deleted: tuple[str, ...]
    waiting: tuple[str, ...]

    @property
    def changed(self) -> bool:
        return bool(self.uploaded or self.deleted)


def _signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def local_files(root: Path) -> dict[str, Path]:
    """Every file under root that belongs in the mirror, by its path in the repo."""
    files = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in SKIPPED_DIRS for part in relative.parts):
            continue
        if not path.is_file() or path.name in SKIPPED_FILES:
            continue
        if path.name.endswith(SKIPPED_SUFFIXES) or ".partial" in path.suffixes:
            continue
        files[relative.as_posix()] = path
    return files


@dataclass
class HubMirror:
    """A local folder kept in a private Hugging Face dataset repository."""

    repo_id: str
    local_dir: Path
    token: str | None = None
    settle_seconds: float = 60.0
    api: object | None = None
    clock: Callable[[], float] = time.time
    _pushed: dict[str, tuple[int, int]] = field(default_factory=dict, init=False, repr=False)
    _restored: bool = field(default=False, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _stop: threading.Event | None = field(default=None, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.local_dir = Path(self.local_dir)
        if self.api is None:
            from huggingface_hub import HfApi

            self.api = HfApi(token=self.token)

    @property
    def url(self) -> str:
        return f"https://huggingface.co/datasets/{self.repo_id}"

    def restore(self) -> int:
        """Create the repository if needed and bring its files down. Returns the count."""
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.api.create_repo(self.repo_id, repo_type="dataset", private=True, exist_ok=True)
        self.api.snapshot_download(
            repo_id=self.repo_id, repo_type="dataset", local_dir=str(self.local_dir)
        )
        files = local_files(self.local_dir)
        self._pushed = {name: _signature(path) for name, path in files.items()}
        self._restored = True
        return len(files)

    def push(self, message: str = "sync from molab") -> PushReport:
        """Send what changed since the last push or the restore."""
        if not self._restored:
            raise RuntimeError("restore() must succeed before the first push")
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete

        with self._lock:
            now = self.clock()
            current = local_files(self.local_dir)
            uploads, waiting, signatures = [], [], {}
            for name, path in current.items():
                try:
                    signature = _signature(path)
                except FileNotFoundError:
                    continue
                if self._pushed.get(name) == signature:
                    continue
                if now - signature[1] / 1e9 < self.settle_seconds:
                    waiting.append(name)
                    continue
                uploads.append(name)
                signatures[name] = signature
            deletes = [name for name in self._pushed if name not in current]

            operations = [
                CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(current[name]))
                for name in uploads
            ] + [CommitOperationDelete(path_in_repo=name) for name in deletes]
            if operations:
                self.api.create_commit(
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    operations=operations,
                    commit_message=message,
                )
            self._pushed.update(signatures)
            for name in deletes:
                self._pushed.pop(name, None)
            return PushReport(tuple(uploads), tuple(deletes), tuple(waiting))

    def push_all(self, message: str = "sync from molab") -> PushReport:
        """Push without waiting for recent files to settle, for the end of a cell."""
        settle, self.settle_seconds = self.settle_seconds, 0.0
        try:
            return self.push(message)
        finally:
            self.settle_seconds = settle

    def start_background(self, interval_seconds: float = 600.0) -> None:
        """Push every interval from a daemon thread until stop_background()."""
        self.stop_background()
        stop = threading.Event()

        def loop() -> None:
            while not stop.wait(interval_seconds):
                try:
                    report = self.push("periodic sync from molab")
                except Exception as error:  # a failed push is retried next interval
                    print(f"[hub sync] push failed, retrying later: {error}")
                    continue
                if report.changed:
                    print(
                        f"[hub sync] {len(report.uploaded)} file(s) sent, "
                        f"{len(report.deleted)} removed"
                    )

        self._stop = stop
        self._thread = threading.Thread(target=loop, name="hub-sync", daemon=True)
        self._thread.start()

    def stop_background(self) -> None:
        if self._stop is not None:
            self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._stop, self._thread = None, None
