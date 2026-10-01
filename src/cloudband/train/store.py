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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from cloudband.train.protocol import LrSearchManifest, LrSearchRun, TrainProtocol


@dataclass(frozen=True)
class RunStore:
    """Where finished runs live, and where fastai writes them while training."""

    results_dir: Path
    checkpoint_dir: Path
    source_dir: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "results_dir", Path(self.results_dir))
        object.__setattr__(self, "checkpoint_dir", Path(self.checkpoint_dir))
        object.__setattr__(self, "source_dir", Path(self.source_dir))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

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