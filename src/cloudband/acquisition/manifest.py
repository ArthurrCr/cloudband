"""Tracking which scenes a collection needs and which are actually present.

A scene that silently fails to download produces a result computed over fewer
scenes than reported, so presence is checked against a declared expectation
rather than inferred from whatever is on disk.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

# A Sentinel-2 SAFE directory is incomplete without these.
SAFE_REQUIRED_ENTRIES: tuple[str, ...] = ("MTD_MSIL1C.xml", "GRANULE")

# Below this a SAFE directory is a stub, not a product.
MIN_SAFE_BYTES = 100 * 1024 * 1024


@dataclass(frozen=True)
class SceneStatus:
    """Whether one expected scene is present and usable."""

    name: str
    path: Path
    present: bool
    complete: bool
    size_bytes: int
    missing_entries: tuple[str, ...] = ()

    @property
    def usable(self) -> bool:
        return self.present and self.complete


def directory_size(path: Path) -> int:
    """Total size in bytes of every file under a directory."""
    if not path.is_dir():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def find_scene(root: Path, name: str) -> Path | None:
    """Locate a SAFE directory by product name, with or without the suffix."""
    for candidate in (root / name, root / f"{name}.SAFE"):
        if candidate.is_dir():
            return candidate
    matches = sorted(root.glob(f"{name}*"))
    return next((match for match in matches if match.is_dir()), None)


def check_scene(root: Path, name: str, min_bytes: int = MIN_SAFE_BYTES) -> SceneStatus:
    """Report whether one scene is present and structurally complete."""
    path = find_scene(root, name)
    if path is None:
        return SceneStatus(name=name, path=root / name, present=False, complete=False, size_bytes=0)

    missing = tuple(entry for entry in SAFE_REQUIRED_ENTRIES if not (path / entry).exists())
    size = directory_size(path)
    return SceneStatus(
        name=name,
        path=path,
        present=True,
        complete=not missing and size >= min_bytes,
        size_bytes=size,
        missing_entries=missing,
    )


def check_collection(
    root: Path, expected: list[str], min_bytes: int = MIN_SAFE_BYTES
) -> list[SceneStatus]:
    """Report on every expected scene, in the order given."""
    return [check_scene(root, name, min_bytes) for name in expected]


def missing_scenes(statuses: list[SceneStatus]) -> list[str]:
    """Names of scenes that are absent or incomplete."""
    return [status.name for status in statuses if not status.usable]


def require_complete(statuses: list[SceneStatus]) -> None:
    """Raise unless every expected scene is present and complete.

    Scoring a partial collection yields a number over fewer scenes than the
    published reference, which no downstream check would catch.
    """
    missing = missing_scenes(statuses)
    if missing:
        raise RuntimeError(
            f"{len(missing)} of {len(statuses)} scenes are missing or incomplete: {missing}"
        )


def summarise(statuses: list[SceneStatus]) -> dict[str, object]:
    """Counts and total size, for printing and for the run manifest."""
    usable = [status for status in statuses if status.usable]
    return {
        "expected": len(statuses),
        "usable": len(usable),
        "missing": missing_scenes(statuses),
        "total_gib": round(sum(status.size_bytes for status in usable) / 1024**3, 2),
    }


def file_digest(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the sha256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()