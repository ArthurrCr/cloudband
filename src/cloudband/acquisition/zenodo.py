"""Fetching the PixBox Sentinel-2 collection from its Zenodo record.

The record holds two archives: a small one with the label CSV and the class
definitions, and a 20 GiB one holding one zipped SAFE product per scene.
Downloads are resumable, and extraction is skipped for products already present.
"""

from __future__ import annotations

import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path

S2_RECORD_ID = "5036991"
L8_RECORD_ID = "5040271"

# Kept for callers written before the Landsat collection was added.
RECORD_ID = S2_RECORD_ID

LABELS_ARCHIVE = "PixBox-S2-CMIX.zip"
SCENES_ARCHIVE = "Sentinel-2_L1C.zip"

LABELS_CSV = "PixBox-S2-CMIX/pixbox_sentinel2_cmix_20180425.csv"
DEFINITIONS = "PixBox-S2-CMIX/pixbox_sentinel2_cmix_20180425_definitions.txt"

L8_LABELS_ARCHIVE = "PixBox-L8-CMIX.zip"
L8_SCENES_ARCHIVE = "Landsat8_L1.zip"
L8_LABELS_CSV = "PixBox-L8-CMIX/pixbox_landsat8_cmix_20150527.csv"

# The Landsat collection names its class table descriptions, not definitions.
L8_DESCRIPTIONS = "PixBox-L8-CMIX/pixbox_landsat8_cmix_20150527_descriptions.txt"

SCENES_PREFIX = "Sentinel-2_L1C/"
INNER_SUFFIX = ".SAFE.zip"

SCENES_ARCHIVE_BYTES = 22000157118
SCENES_IN_ARCHIVE = 28
L8_SCENES_IN_ARCHIVE = 11


@dataclass(frozen=True)
class Archive:
    """One downloadable file of a Zenodo record."""

    filename: str
    record_id: str = S2_RECORD_ID
    expected_bytes: int | None = None

    @property
    def url(self) -> str:
        return f"https://zenodo.org/records/{self.record_id}/files/{self.filename}?download=1"


LABELS = Archive(LABELS_ARCHIVE, S2_RECORD_ID)
SCENES = Archive(SCENES_ARCHIVE, S2_RECORD_ID, expected_bytes=SCENES_ARCHIVE_BYTES)

L8_LABELS = Archive(L8_LABELS_ARCHIVE, L8_RECORD_ID)
L8_SCENES = Archive(L8_SCENES_ARCHIVE, L8_RECORD_ID)


def download(archive: Archive, target_dir: Path, connections: int = 16) -> Path:
    """Fetch an archive, resuming an interrupted transfer when possible.

    aria2c is used when available because a 20 GiB transfer over a single
    connection fails often enough to matter. It resumes rather than restarts.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = target_dir / archive.filename

    if is_complete(archive, target_dir):
        return destination

    command = _download_command(archive, target_dir, connections)
    subprocess.run(command, check=True)

    if not is_complete(archive, target_dir):
        actual = destination.stat().st_size if destination.exists() else 0
        raise RuntimeError(
            f"{archive.filename} is {actual} bytes, expected {archive.expected_bytes}"
        )
    return destination


def _download_command(archive: Archive, target_dir: Path, connections: int) -> list[str]:
    if _has_aria2():
        return [
            "aria2c",
            f"-x{connections}",
            f"-s{connections}",
            "-k1M",
            "--continue=true",
            "--file-allocation=none",
            "--summary-interval=30",
            "-d",
            str(target_dir),
            "-o",
            archive.filename,
            archive.url,
        ]
    return ["curl", "-L", "-C", "-", "-o", str(target_dir / archive.filename), archive.url]


def _has_aria2() -> bool:
    return subprocess.run(["which", "aria2c"], capture_output=True, check=False).returncode == 0


def is_complete(archive: Archive, target_dir: Path) -> bool:
    """Whether a downloaded archive is present and of the expected size."""
    path = Path(target_dir) / archive.filename
    if not path.is_file():
        return False
    if archive.expected_bytes is None:
        return path.stat().st_size > 0
    return path.stat().st_size == archive.expected_bytes


def inner_product_names(archive_path: Path) -> list[str]:
    """Product names of the zipped SAFE directories inside the scenes archive."""
    with zipfile.ZipFile(archive_path) as handle:
        entries = [name for name in handle.namelist() if name.endswith(INNER_SUFFIX)]
    return sorted(Path(name).name.removesuffix(INNER_SUFFIX) for name in entries)


def extract_scenes(
    archive_path: Path, target_dir: Path, keep_inner_zip: bool = False
) -> list[Path]:
    """Unpack every SAFE product, skipping those already extracted.

    The outer archive holds one zip per scene, so each is extracted twice: once
    out of the outer archive and once into a SAFE directory.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []

    with zipfile.ZipFile(archive_path) as outer:
        entries = [name for name in outer.namelist() if name.endswith(INNER_SUFFIX)]
        for entry in sorted(entries):
            product = Path(entry).name.removesuffix(INNER_SUFFIX)
            safe_dir = target_dir / f"{product}.SAFE"
            if safe_dir.is_dir():
                extracted.append(safe_dir)
                continue

            inner_path = target_dir / Path(entry).name
            with outer.open(entry) as source, inner_path.open("wb") as sink:
                sink.write(source.read())
            with zipfile.ZipFile(inner_path) as inner:
                inner.extractall(target_dir)
            if not keep_inner_zip:
                inner_path.unlink()
            extracted.append(safe_dir)

    return extracted