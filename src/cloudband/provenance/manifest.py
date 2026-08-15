"""Provenance record written next to every result."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def git_commit(repository: Path | None = None) -> str:
    """Return the current commit hash, or unknown outside a repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository or Path.cwd(),
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return result.stdout.strip()


def file_digest(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the sha256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Manifest:
    """Everything needed to reproduce one result."""

    result_name: str
    dataset: str
    model_id: str
    commit: str
    python_version: str
    platform_name: str
    created_at: str
    seed: int | None = None
    package_versions: dict[str, str] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    inputs: dict[str, str] = field(default_factory=dict)

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")
        return path


def build_manifest(
    result_name: str,
    dataset: str,
    model_id: str,
    seed: int | None = None,
    package_versions: dict[str, str] | None = None,
    config: dict[str, Any] | None = None,
    inputs: dict[str, Path] | None = None,
) -> Manifest:
    """Collect the current environment into a manifest."""
    return Manifest(
        result_name=result_name,
        dataset=dataset,
        model_id=model_id,
        commit=git_commit(),
        python_version=sys.version.split()[0],
        platform_name=platform.platform(),
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        seed=seed,
        package_versions=package_versions or {},
        config=config or {},
        inputs={name: file_digest(path) for name, path in (inputs or {}).items()},
    )