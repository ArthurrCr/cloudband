"""Contract tests for scene presence checking.

A scene that fails to download must be detected before scoring, not inferred
afterwards from a total that nobody checked.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cloudband.acquisition import manifest
from cloudband.labels.pixbox_s2 import PRODUCT_ID_TO_SCENE

SMALL = 1024


def make_safe(root: Path, name: str, complete: bool = True, size: int = SMALL) -> Path:
    path = root / f"{name}.SAFE"
    (path / "GRANULE").mkdir(parents=True)
    if complete:
        (path / "MTD_MSIL1C.xml").write_bytes(b"x" * size)
    return path


def test_expected_scene_count_matches_the_collection() -> None:
    assert len(PRODUCT_ID_TO_SCENE) == 29


def test_complete_scene_is_usable(tmp_path: Path) -> None:
    make_safe(tmp_path, "S2A_MSIL1C_TEST")
    status = manifest.check_scene(tmp_path, "S2A_MSIL1C_TEST", min_bytes=1)
    assert status.usable
    assert status.missing_entries == ()


def test_absent_scene_is_reported(tmp_path: Path) -> None:
    status = manifest.check_scene(tmp_path, "S2A_MSIL1C_ABSENT")
    assert not status.present
    assert not status.usable
    assert status.size_bytes == 0


def test_scene_missing_metadata_is_incomplete(tmp_path: Path) -> None:
    make_safe(tmp_path, "S2A_MSIL1C_TEST", complete=False)
    status = manifest.check_scene(tmp_path, "S2A_MSIL1C_TEST", min_bytes=1)
    assert status.present
    assert not status.complete
    assert "MTD_MSIL1C.xml" in status.missing_entries


def test_truncated_download_is_incomplete(tmp_path: Path) -> None:
    """A directory that exists but holds almost nothing is a failed download."""
    make_safe(tmp_path, "S2A_MSIL1C_TEST", size=10)
    status = manifest.check_scene(tmp_path, "S2A_MSIL1C_TEST", min_bytes=SMALL)
    assert status.present
    assert not status.complete
    assert status.missing_entries == ()


def test_scene_found_without_safe_suffix(tmp_path: Path) -> None:
    path = tmp_path / "S2A_MSIL1C_TEST"
    (path / "GRANULE").mkdir(parents=True)
    (path / "MTD_MSIL1C.xml").write_bytes(b"x" * SMALL)
    assert manifest.check_scene(tmp_path, "S2A_MSIL1C_TEST", min_bytes=1).usable


def test_collection_reports_every_expected_scene(tmp_path: Path) -> None:
    make_safe(tmp_path, "scene_a")
    statuses = manifest.check_collection(tmp_path, ["scene_a", "scene_b"], min_bytes=1)
    assert len(statuses) == 2
    assert manifest.missing_scenes(statuses) == ["scene_b"]


def test_require_complete_raises_on_partial_collection(tmp_path: Path) -> None:
    make_safe(tmp_path, "scene_a")
    statuses = manifest.check_collection(tmp_path, ["scene_a", "scene_b"], min_bytes=1)
    with pytest.raises(RuntimeError, match="1 of 2 scenes are missing"):
        manifest.require_complete(statuses)


def test_require_complete_passes_on_full_collection(tmp_path: Path) -> None:
    make_safe(tmp_path, "scene_a")
    make_safe(tmp_path, "scene_b")
    statuses = manifest.check_collection(tmp_path, ["scene_a", "scene_b"], min_bytes=1)
    manifest.require_complete(statuses)


def test_summary_counts_and_size(tmp_path: Path) -> None:
    make_safe(tmp_path, "scene_a")
    statuses = manifest.check_collection(tmp_path, ["scene_a", "scene_b"], min_bytes=1)
    summary = manifest.summarise(statuses)
    assert summary["expected"] == 2
    assert summary["usable"] == 1
    assert summary["missing"] == ["scene_b"]


def test_digest_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "file.bin"
    path.write_bytes(b"content")
    assert manifest.file_digest(path) == manifest.file_digest(path)