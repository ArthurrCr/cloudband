"""Persist training runs one at a time, so a session can die and resume.

A full comparison is hundreds of epochs. Nothing it produces may live only in
memory or on the session's local disk: every finished run is written to durable
storage as soon as it ends, and a restart skips what is already there.

A run counts as finished only when its manifest exists, and the manifest is
written after the checkpoint is copied. A copy interrupted halfway therefore
leaves no manifest, and the run is trained again instead of being trusted.
"""

from __future__ import annotations

import errno
import json
import os
import shutil
import tempfile
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch

from cloudband.train.protocol import LrSearchManifest, LrSearchRun, TrainProtocol


# A mounted Drive that sat idle answers its first access with one of these, and the
# next access works. They also come back, for a while, for an entry trashed from
# outside the mount.
DRIVE_HICCUPS = frozenset({errno.ENOTCONN, errno.ESTALE})
RETRY_WAITS = (1.0, 2.0, 4.0, 8.0, 15.0)
_sleep = time.sleep


class DriveDisconnected(OSError):
    """The Drive mount stopped answering, not just one file on it."""


def _listable(folder: Path) -> bool:
    try:
        os.listdir(folder)
    except OSError:
        return False
    return True


def _drive_retry(action, path: Path, absent):
    """Run a read on the Drive, waiting out the errors of a mount that was idle.

    If the error outlasts every wait, the folder decides what it was: a folder that
    still lists means only this entry is unreadable, a ghost of a file trashed from
    outside, and `absent()` answers for it; a folder that does not means the mount is
    gone, and guessing an answer could hand a run that another session holds to this
    one, so it stops instead.
    """
    last = None
    for wait in (*RETRY_WAITS, None):
        try:
            return action()
        except OSError as error:
            if error.errno not in DRIVE_HICCUPS:
                raise
            last = error
            if wait is not None:
                _sleep(wait)
    if _listable(Path(path).parent):
        return absent()
    raise DriveDisconnected(
        last.errno,
        f"the Drive mount is not answering ({last.strerror}). Run "
        'drive.mount("/content/drive", force_remount=True) and then this cell again',
        str(path),
    ) from last


def _ghost(path: Path):
    def absent():
        raise FileNotFoundError(errno.ENOENT, "listed by the Drive but not readable", str(path))

    return absent


def _exists(path: Path) -> bool:
    return _drive_retry(path.is_file, path, absent=lambda: False)


def _read_text(path: Path) -> str:
    return _drive_retry(path.read_text, path, absent=_ghost(path))


def _is_file(path: Path) -> bool:
    """is_file() that reads a Drive entry which cannot be read as absent.

    Right after files are trashed from outside the mount, the listing can still show
    them for a while, and asking about one fails with "Transport endpoint is not
    connected". For listing and reporting that means "not there", at once; the reads
    that train go through _exists, which first waits for the mount to answer.
    """
    try:
        return path.is_file()
    except OSError:
        return False


@dataclass(frozen=True)
class RunStore:
    """Where finished runs live, and where fastai writes them while training."""

    results_dir: Path
    checkpoint_dir: Path
    source_dir: Path
    progress_dir: Path | None = None
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    # the locks this session holds: name -> (when it started, when it last wrote).
    # Kept in memory because a mounted Drive can return an old copy of a file that
    # was just replaced, so what this session wrote must not depend on reading it back.
    _held: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "results_dir", Path(self.results_dir))
        object.__setattr__(self, "checkpoint_dir", Path(self.checkpoint_dir))
        object.__setattr__(self, "source_dir", Path(self.source_dir))
        progress = self.progress_dir or self.results_dir / "progress"
        object.__setattr__(self, "progress_dir", Path(progress))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.progress_dir.mkdir(parents=True, exist_ok=True)

    def wake(self) -> None:
        """Make the mount answer before the work that follows reads it.

        A Drive mount left idle, for instance while the local cache is copied, refuses
        its first access; listing the folders here takes that refusal, with the same
        waits as every read, so the cell that comes next starts on a mount that answers.
        """
        for folder in (self.results_dir, self.checkpoint_dir, self.progress_dir):
            _drive_retry(lambda folder=folder: os.listdir(folder), folder, absent=lambda: None)

    def largest_files(self, root: Path, count: int = 15, exclude=()) -> list[tuple[Path, int]]:
        """The biggest files under `root`, skipping entries the Drive lists but cannot read."""
        self.wake()
        excluded = set(exclude)
        sizes = []
        for path in Path(root).rglob("*"):
            if path in excluded:
                continue
            try:
                if path.is_file():
                    sizes.append((path, path.stat().st_size))
            except OSError:
                continue
        return sorted(sizes, key=lambda item: item[1], reverse=True)[:count]

    def manifest_path(self, result_name: str) -> Path:
        return self.results_dir / f"{result_name}.json"

    def checkpoint_path(self, checkpoint_name: str) -> Path:
        return self.checkpoint_dir / f"{checkpoint_name}.pth"

    def is_done(self, checkpoint_name: str) -> bool:
        """True when the run was fully saved, checkpoint and manifest both."""
        return _exists(self.manifest_path(checkpoint_name)) and _exists(
            self.checkpoint_path(checkpoint_name)
        )

    def save_run(self, run) -> tuple[Path, Path]:
        """Copy the checkpoint, then write the manifest that marks it done."""
        name = run.fit_result.checkpoint_name
        source = self.source_dir / f"{name}.pth"
        if not source.is_file():
            raise FileNotFoundError(f"no checkpoint to save at {source}")

        target = self.checkpoint_path(name)
        partial = target.with_suffix(".pth.partial")
        shutil.copyfile(source, partial)
        partial.replace(target)

        manifest = run.manifest.write(self.manifest_path(name))
        return manifest, target

    # ---- progress of a run in flight, saved after every epoch ----

    def epoch_state_path(self, name: str) -> Path:
        return self.progress_dir / f"{name}.state.pt"

    def save_epoch_state(self, name: str, state: dict) -> Path:
        """Write the state through a local file, then swap it in whole.

        Torch writes in several passes, which a mounted Drive handles badly, so the
        file is finished on the local disk and copied once. The previous state stays
        in place until the new one is complete.
        """
        import torch

        path = self.epoch_state_path(name)
        partial = path.with_suffix(".pt.partial")
        with tempfile.TemporaryDirectory() as scratch:
            local = Path(scratch) / "state.pt"
            torch.save(state, local)
            shutil.copyfile(local, partial)
        partial.replace(path)
        return path

    def load_epoch_state(self, name: str) -> dict | None:
        import torch

        path = self.epoch_state_path(name)
        if not _exists(path):
            return None
        return _drive_retry(
            lambda: torch.load(path, map_location="cpu", weights_only=False),
            path,
            absent=_ghost(path),
        )

    def clear_epoch_state(self, name: str) -> None:
        self.epoch_state_path(name).unlink(missing_ok=True)

    def discard(self, name: str) -> list[Path]:
        """Delete everything a run left: manifest, checkpoint, progress and lock.

        For a run that must be trained again, such as one made at a learning rate
        other than the one its search chose. Returns the files that were removed.
        """
        candidates = [
            self.manifest_path(name),
            self.checkpoint_path(name),
            self.checkpoint_path(name).with_suffix(".pth.partial"),
            self.epoch_state_path(name),
            self.epoch_state_path(name).with_suffix(".pt.partial"),
            self.lock_path(name),
        ]
        removed = [path for path in candidates if path.is_file()]
        for path in removed:
            path.unlink()
        self._held.pop(name, None)
        return removed

    def unused_files(
        self,
        now: float | None = None,
        lease_seconds: float = 3600.0,
        min_age: float = 3600.0,
        skipped: list[Path] | None = None,
    ) -> list[tuple[Path, str]]:
        """Files in the shared folder that nothing will read again, each with its reason.

        Only leftovers that are certain: copies interrupted halfway (".partial"), probes
        of the folder check, progress or locks of a run that is already done, locks that
        nobody renews and that have no progress to continue, and a checkpoint or
        manifest without its other half. A run with a live lock, or any file touched in
        the last `min_age` seconds, is never listed, since a session may be writing it.
        Progress that a session can still continue is never listed either. This only
        lists; remove_unused deletes. Entries of the Drive that cannot be read (see _is_file)
        are left out, and added to `skipped` when it is given.
        """
        self.wake()
        now = time.time() if now is None else now
        found: list[tuple[Path, str]] = []
        live: set[str] = set()
        for lock in self.progress_dir.glob("*.lock"):
            info = self.lock_info(lock.name[: -len(".lock")])
            if info is not None and now - info["heartbeat"] < lease_seconds:
                live.add(lock.name[: -len(".lock")])

        def add(path: Path, reason: str) -> None:
            try:
                if not path.is_file():
                    return
                recent = now - path.stat().st_mtime < min_age
            except OSError:
                if skipped is not None:
                    skipped.append(path)
                return
            if not recent and path not in {p for p, _ in found}:
                found.append((path, reason))

        def done(name: str) -> bool:
            return _is_file(self.manifest_path(name)) and _is_file(self.checkpoint_path(name))

        for folder in {self.results_dir, self.checkpoint_dir, self.progress_dir}:
            for path in folder.glob("*.partial"):
                add(path, "a copy that was interrupted halfway")
            for path in folder.glob(".write-test-*"):
                add(path, "a probe left by the folder check")

        for lock in self.progress_dir.glob("*.lock"):
            name = lock.name[: -len(".lock")]
            if name in live:
                continue
            if done(name):
                add(lock, "lock of a run that is already done")
            elif not _is_file(self.epoch_state_path(name)):
                add(lock, "stale lock of a run with no progress to continue")

        for state in self.progress_dir.glob("*.state.pt"):
            name = state.name[: -len(".state.pt")]
            if name not in live and done(name):
                add(state, "progress of a run that is already done")

        for checkpoint in self.checkpoint_dir.glob("*.pth"):
            name = checkpoint.stem
            if name not in live and not _is_file(self.manifest_path(name)):
                add(checkpoint, "a checkpoint with no manifest: the run was never marked done")

        for manifest in self.results_dir.glob("*.json"):
            name = manifest.stem
            if name == "settings" or name.endswith(
                ("-lr-search", "-lr-search.partial", "-lr-curves")
            ):
                continue
            if name not in live and not _is_file(self.checkpoint_path(name)):
                add(manifest, "a manifest with no checkpoint: the run cannot be used")
        return sorted(found)

    def remove_unused(self, items: list[tuple[Path, str]]) -> int:
        """Delete the files unused_files listed. Returns how many were removed."""
        removed = 0
        for path, _ in items:
            if _is_file(path):
                path.unlink()
                removed += 1
        return removed

    # ---- the learning-rate search, kept candidate by candidate ----

    def search_progress_path(self, protocol: TrainProtocol) -> Path:
        return self.results_dir / f"{protocol.run_id}-lr-search.partial.json"

    def save_search_run(
        self, protocol: TrainProtocol, run: LrSearchRun, budget: tuple, seed: int
    ) -> None:
        """Keep one finished candidate, so a stopped search never repeats it."""
        path = self.search_progress_path(protocol)
        records = json.loads(_read_text(path)) if _exists(path) else []
        records = [r for r in records if r["learning_rate"] != run.learning_rate]
        records.append(
            {
                "run_id": run.run_id,
                "learning_rate": run.learning_rate,
                "val_loss": run.val_loss,
                "split": run.split,
                "budget": list(budget),
                "seed": seed,
            }
        )
        partial = path.with_suffix(".json.partial")
        partial.write_text(json.dumps(records, indent=2))
        partial.replace(path)

    def load_search_run(
        self, protocol: TrainProtocol, learning_rate: float, budget: tuple, seed: int
    ) -> LrSearchRun | None:
        path = self.search_progress_path(protocol)
        if not _exists(path):
            return None
        for record in json.loads(_read_text(path)):
            if record["learning_rate"] != learning_rate:
                continue
            if record["budget"] != list(budget) or record["seed"] != seed:
                raise RuntimeError(
                    f"{protocol.run_id}: the saved search result for {learning_rate:g} "
                    f"was made with budget {record['budget']} and seed "
                    f"{record['seed']}, not {list(budget)} and {seed}; delete "
                    f"{path.name} to search again"
                )
            return LrSearchRun(
                run_id=record["run_id"],
                learning_rate=record["learning_rate"],
                val_loss=record["val_loss"],
                split=record["split"],
            )
        return None

    # ---- one session at a time on a run ----

    def lock_path(self, name: str) -> Path:
        return self.progress_dir / f"{name}.lock"

    def lock_info(self, name: str) -> dict | None:
        path = self.lock_path(name)
        if not _exists(path):
            return None
        try:
            return json.loads(_read_text(path))
        except (json.JSONDecodeError, FileNotFoundError):
            return None

    def _write_lock(self, name: str, started: float | None = None) -> float:
        now = time.time()
        record = {
            "owner": self.session_id,
            "started": started or now,
            "heartbeat": now,
        }
        path = self.lock_path(name)
        partial = path.with_suffix(".lock.partial")
        partial.write_text(json.dumps(record))
        partial.replace(path)
        self._held[name] = (record["started"], now)
        return now

    def _confirmed(
        self,
        name: str,
        written_at: float,
        attempts: int,
        poll_seconds: float,
        sleep: Callable[[float], None],
    ) -> bool:
        """Did the lock just written stay this session's?

        A mounted Drive can answer with the file as it was before the write, or with
        nothing, for a while after it was replaced. That old copy is no proof that
        another session took the run: only a lock written by another session at or
        after this write is. So an old or unreadable copy is read again, and if
        nothing newer from another session ever shows up, the run is this session's.
        """
        for attempt in range(attempts):
            info = self.lock_info(name)
            if info is not None:
                if info["owner"] == self.session_id:
                    return True
                if info["heartbeat"] >= written_at:
                    return False
            if attempt + 1 < attempts:
                sleep(poll_seconds)
        return True

    def claim(
        self,
        name: str,
        lease_seconds: float = 3600.0,
        break_locks: bool = False,
        confirm_seconds: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
        attempts: int = 8,
    ) -> bool:
        """Take a run for this session. False when another session has it.

        A run belongs to the session that last wrote its lock, for as long as that
        session keeps sending heartbeats, one per epoch. A lock nobody has renewed
        for lease_seconds is stale and is taken over. break_locks takes over any
        lock, for when it is known that the other session is gone. Drive cannot
        make this atomic across machines, so after writing the lock it is read back
        to see that it is still this session's, allowing the Drive a few attempts to
        show the new file (see _confirmed).
        """
        info = self.lock_info(name)
        held_by_another = (
            info is not None and info["owner"] != self.session_id and not break_locks
        )
        if held_by_another and time.time() - info["heartbeat"] < lease_seconds:
            return False
        written_at = self._write_lock(name)
        sleep(confirm_seconds)
        if self._confirmed(name, written_at, attempts, confirm_seconds, sleep):
            return True
        self._held.pop(name, None)
        return False

    def heartbeat(self, name: str) -> None:
        held = self._held.get(name)
        if held is None:
            return
        started, written_at = held
        info = self.lock_info(name)
        if (
            info is not None
            and info["owner"] != self.session_id
            and info["heartbeat"] >= written_at
        ):
            return                       # another session took the run over
        self._write_lock(name, started=started)

    def release(self, name: str) -> None:
        held = self._held.pop(name, None)
        info = self.lock_info(name)
        if held is not None:
            _, written_at = held
            taken_over = (
                info is not None
                and info["owner"] != self.session_id
                and info["heartbeat"] >= written_at
            )
            if not taken_over:
                self.lock_path(name).unlink(missing_ok=True)
            return
        if info is not None and info["owner"] == self.session_id:
            self.lock_path(name).unlink(missing_ok=True)

    def describe_lock(self, name: str) -> str:
        info = self.lock_info(name)
        if info is None:
            return "no lock"
        minutes = (time.time() - info["heartbeat"]) / 60
        return f"session {info['owner']}, last heartbeat {minutes:.0f} min ago"

    # ---- choices made once for the whole experiment ----

    def settings_path(self) -> Path:
        return self.results_dir / "settings.json"

    def load_setting(self, key: str, default=None):
        path = self.settings_path()
        if not _exists(path):
            return default
        return json.loads(_read_text(path)).get(key, default)

    def save_setting(self, key: str, value) -> None:
        path = self.settings_path()
        settings = json.loads(_read_text(path)) if _exists(path) else {}
        settings[key] = value
        partial = path.with_suffix(".json.partial")
        partial.write_text(json.dumps(settings, indent=2))
        partial.replace(path)

    def convergence(self, checkpoint_name: str, window: int = 3) -> dict | None:
        """Where the best checkpoint fell, and whether training was still improving.

        A best epoch inside the last `window` epochs means the validation loss
        was still falling when training stopped, so the budget was probably too
        short. Returns None when the run has no recorded history.
        """
        path = self.manifest_path(checkpoint_name)
        if not _exists(path):
            return None
        config = json.loads(_read_text(path)).get("config", {})
        history = config.get("valid_loss_history") or []
        if not history:
            return None
        best = config["best_epoch"]
        return {
            "run": checkpoint_name,
            "epochs": len(history),
            "best_epoch": best,
            "best_val_loss": history[best],
            "still_improving": best >= len(history) - window,
        }

    def curves_path(self, protocol: TrainProtocol) -> Path:
        return self.results_dir / f"{protocol.run_id}-lr-curves.json"

    def load_curves(self, protocol: TrainProtocol) -> dict[float, list[float]]:
        """The validation loss after every epoch of each candidate that has it saved."""
        path = self.curves_path(protocol)
        if not _exists(path):
            return {}
        return {
            float(rate): list(curve) for rate, curve in json.loads(_read_text(path)).items()
        }

    def save_curve(self, protocol: TrainProtocol, learning_rate: float, curve) -> None:
        """Keep the validation curve of one candidate, next to the search records.

        The search record only has the best loss of a candidate; its curve is what shows
        the proxy budget was enough, and it would otherwise live only in progress files
        that are deleted once the candidate ends.
        """
        curves = self.load_curves(protocol)
        curves[float(learning_rate)] = [float(value) for value in curve]
        path = self.curves_path(protocol)
        partial = path.with_suffix(".json.partial")
        partial.write_text(json.dumps({f"{r:g}": c for r, c in sorted(curves.items())}, indent=2))
        partial.replace(path)

    def search_path(self, protocol: TrainProtocol) -> Path:
        return self.results_dir / f"{protocol.run_id}-lr-search.json"

    def save_search(self, protocol: TrainProtocol, search: LrSearchManifest) -> Path:
        payload = {
            "architecture": search.architecture,
            "grid": list(search.grid),
            "runs": [
                {
                    "run_id": run.run_id,
                    "learning_rate": run.learning_rate,
                    "val_loss": run.val_loss,
                    "split": run.split,
                }
                for run in search.runs
            ],
        }
        path = self.search_path(protocol)
        partial = path.with_suffix(".json.partial")
        partial.write_text(json.dumps(payload, indent=2))
        partial.replace(path)
        return path

    def load_search(self, protocol: TrainProtocol) -> LrSearchManifest | None:
        path = self.search_path(protocol)
        if not _exists(path):
            return None
        payload = json.loads(_read_text(path))
        return LrSearchManifest(
            architecture=payload["architecture"],
            runs=tuple(LrSearchRun(**run) for run in payload["runs"]),
            grid=tuple(payload["grid"]),
        )


def load_model_from_checkpoint(
    model_builder: Callable[[], torch.nn.Module],
    path: Path,
    device: str | torch.device = "cpu",
) -> torch.nn.Module:
    """Rebuild a model and load a checkpoint written by fit_protocol.

    fastai saves a bare state dict when the optimiser is not included, which is
    how fit_protocol saves. model_builder should build the untrained
    architecture; its pretrained weights are overwritten by the checkpoint.
    """
    model = model_builder()
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def ensure_shared_folder(folder: Path) -> Path:
    """Check that the shared folder is there and can be written to.

    Every session must read and write the same files, so a missing folder is an
    error that says how to fix it, and not a new empty folder in the wrong Drive.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(
            f"the shared folder {folder} was not found. Create it once in the Drive "
            "that owns it and share it, as editor, with every account that trains. "
            "In each of those accounts, open Drive > Shared with me, right-click the "
            "folder and choose 'Add shortcut to Drive' under My Drive, keeping this "
            "same name. Then run this cell again."
        )
    probe = folder / f".write-test-{uuid.uuid4().hex[:8]}"
    try:
        probe.write_text("ok")
        probe.unlink()
    except OSError as error:
        raise PermissionError(
            f"the shared folder {folder} exists but cannot be written to ({error}); "
            "it must be shared with this account as editor, not as viewer"
        ) from error
    return folder