"""Persist training runs one at a time, so a session can die and resume.

A full comparison is hundreds of epochs. Nothing it produces may live only in
memory or on the session's local disk: every finished run is written to durable
storage as soon as it ends, and a restart skips what is already there.

A run counts as finished only when its manifest exists, and the manifest is
written after the checkpoint is copied. A copy interrupted halfway therefore
leaves no manifest, and the run is trained again instead of being trusted.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch

from cloudband.train.protocol import LrSearchManifest, LrSearchRun, TrainProtocol


@dataclass(frozen=True)
class RunStore:
    """Where finished runs live, and where fastai writes them while training."""

    results_dir: Path
    checkpoint_dir: Path
    source_dir: Path
    progress_dir: Path | None = None
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def __post_init__(self) -> None:
        object.__setattr__(self, "results_dir", Path(self.results_dir))
        object.__setattr__(self, "checkpoint_dir", Path(self.checkpoint_dir))
        object.__setattr__(self, "source_dir", Path(self.source_dir))
        progress = self.progress_dir or self.results_dir / "progress"
        object.__setattr__(self, "progress_dir", Path(progress))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.progress_dir.mkdir(parents=True, exist_ok=True)

    def manifest_path(self, result_name: str) -> Path:
        return self.results_dir / f"{result_name}.json"

    def checkpoint_path(self, checkpoint_name: str) -> Path:
        return self.checkpoint_dir / f"{checkpoint_name}.pth"

    def is_done(self, checkpoint_name: str) -> bool:
        """True when the run was fully saved, checkpoint and manifest both."""
        return (
            self.manifest_path(checkpoint_name).is_file()
            and self.checkpoint_path(checkpoint_name).is_file()
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
        if not path.is_file():
            return None
        return torch.load(path, map_location="cpu", weights_only=False)

    def clear_epoch_state(self, name: str) -> None:
        self.epoch_state_path(name).unlink(missing_ok=True)

    # ---- the learning-rate search, kept candidate by candidate ----

    def search_progress_path(self, protocol: TrainProtocol) -> Path:
        return self.results_dir / f"{protocol.run_id}-lr-search.partial.json"

    def save_search_run(
        self, protocol: TrainProtocol, run: LrSearchRun, budget: tuple, seed: int
    ) -> None:
        """Keep one finished candidate, so a stopped search never repeats it."""
        path = self.search_progress_path(protocol)
        records = json.loads(path.read_text()) if path.is_file() else []
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
        if not path.is_file():
            return None
        for record in json.loads(path.read_text()):
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
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def _write_lock(self, name: str, started: float | None = None) -> None:
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

    def claim(
        self,
        name: str,
        lease_seconds: float = 3600.0,
        break_locks: bool = False,
        confirm_seconds: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> bool:
        """Take a run for this session. False when another session has it.

        A run belongs to the session that last wrote its lock, for as long as that
        session keeps sending heartbeats, one per epoch. A lock nobody has renewed
        for lease_seconds is stale and is taken over. break_locks takes over any
        lock, for when it is known that the other session is gone. Drive cannot
        make this atomic across machines, so after writing the lock it is read back
        once to see that it is still this session's.
        """
        info = self.lock_info(name)
        held_by_another = (
            info is not None and info["owner"] != self.session_id and not break_locks
        )
        if held_by_another and time.time() - info["heartbeat"] < lease_seconds:
            return False
        self._write_lock(name)
        sleep(confirm_seconds)
        confirmed = self.lock_info(name)
        return confirmed is not None and confirmed["owner"] == self.session_id

    def heartbeat(self, name: str) -> None:
        info = self.lock_info(name)
        if info is not None and info["owner"] == self.session_id:
            self._write_lock(name, started=info["started"])

    def release(self, name: str) -> None:
        info = self.lock_info(name)
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
        if not path.is_file():
            return default
        return json.loads(path.read_text()).get(key, default)

    def save_setting(self, key: str, value) -> None:
        path = self.settings_path()
        settings = json.loads(path.read_text()) if path.is_file() else {}
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
        if not path.is_file():
            return None
        config = json.loads(path.read_text()).get("config", {})
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
        if not path.is_file():
            return None
        payload = json.loads(path.read_text())
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