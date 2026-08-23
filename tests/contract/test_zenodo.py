"""Contract tests for fetching and unpacking the PixBox Sentinel-2 archives."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from cloudband.acquisition import zenodo

PRODUCT_A = "S2A_MSIL1C_20170113T072241_N0204_R006_T40UEE_20170113T072238"
PRODUCT_B = "S2B_MSIL1C_20170731T102019_N0205_R065_T33VVE_20170731T102348"


def make_inner_zip(path: Path, product: str) -> None:
    """Write a zip holding a minimal SAFE directory."""
    with zipfile.ZipFile(path, "w") as handle:
        handle.writestr(f"{product}.SAFE/MTD_MSIL1C.xml", "<xml/>")
        handle.writestr(f"{product}.SAFE/GRANULE/placeholder", "")


def make_outer_zip(path: Path, products: list[str], tmp_path: Path) -> None:
    """Write the nested archive layout the record publishes."""
    with zipfile.ZipFile(path, "w") as outer:
        for product in products:
            inner = tmp_path / f"{product}.SAFE.zip"
            make_inner_zip(inner, product)
            outer.write(inner, f"{zenodo.SCENES_PREFIX}{product}.SAFE.zip")
            inner.unlink()


def test_archive_urls_point_at_the_record() -> None:
    assert zenodo.RECORD_ID in zenodo.SCENES.url
    assert zenodo.SCENES.url.endswith("Sentinel-2_L1C.zip?download=1")
    assert zenodo.LABELS.url.endswith("PixBox-S2-CMIX.zip?download=1")


def test_archive_holds_28_of_the_29_declared_scenes() -> None:
    """One declared product is absent from the archive. See DIVERGENCES D11."""
    from cloudband.labels.pixbox import PRODUCT_ID_TO_SCENE, UNAVAILABLE_PRODUCT_IDS

    declared = len(PRODUCT_ID_TO_SCENE)
    unavailable = len(UNAVAILABLE_PRODUCT_IDS)
    assert zenodo.SCENES_IN_ARCHIVE + unavailable == declared


def test_incomplete_download_is_detected(tmp_path: Path) -> None:
    (tmp_path / zenodo.SCENES_ARCHIVE).write_bytes(b"truncated")
    assert not zenodo.is_complete(zenodo.SCENES, tmp_path)


def test_absent_download_is_detected(tmp_path: Path) -> None:
    assert not zenodo.is_complete(zenodo.SCENES, tmp_path)


def test_sized_archive_is_complete(tmp_path: Path) -> None:
    archive = zenodo.Archive("small.zip", expected_bytes=4)
    (tmp_path / "small.zip").write_bytes(b"1234")
    assert zenodo.is_complete(archive, tmp_path)


def test_unsized_archive_needs_only_content(tmp_path: Path) -> None:
    (tmp_path / zenodo.LABELS_ARCHIVE).write_bytes(b"x")
    assert zenodo.is_complete(zenodo.LABELS, tmp_path)


def test_inner_product_names_are_listed(tmp_path: Path) -> None:
    archive = tmp_path / "outer.zip"
    make_outer_zip(archive, [PRODUCT_B, PRODUCT_A], tmp_path)
    assert zenodo.inner_product_names(archive) == sorted([PRODUCT_A, PRODUCT_B])


def test_extraction_produces_safe_directories(tmp_path: Path) -> None:
    archive = tmp_path / "outer.zip"
    make_outer_zip(archive, [PRODUCT_A], tmp_path)
    target = tmp_path / "scenes"

    extracted = zenodo.extract_scenes(archive, target)
    assert len(extracted) == 1
    safe_dir = target / f"{PRODUCT_A}.SAFE"
    assert safe_dir.is_dir()
    assert (safe_dir / "MTD_MSIL1C.xml").is_file()
    assert (safe_dir / "GRANULE").is_dir()


def test_inner_zip_is_removed_by_default(tmp_path: Path) -> None:
    archive = tmp_path / "outer.zip"
    make_outer_zip(archive, [PRODUCT_A], tmp_path)
    target = tmp_path / "scenes"

    zenodo.extract_scenes(archive, target)
    assert not list(target.glob("*.SAFE.zip"))


def test_extraction_skips_products_already_present(tmp_path: Path) -> None:
    """Re-running after an interrupted session must not redo finished work."""
    archive = tmp_path / "outer.zip"
    make_outer_zip(archive, [PRODUCT_A], tmp_path)
    target = tmp_path / "scenes"

    zenodo.extract_scenes(archive, target)
    marker = target / f"{PRODUCT_A}.SAFE" / "marker"
    marker.write_text("kept")

    zenodo.extract_scenes(archive, target)
    assert marker.read_text() == "kept"


def test_extracted_scenes_pass_the_presence_check(tmp_path: Path) -> None:
    from cloudband.acquisition import manifest

    archive = tmp_path / "outer.zip"
    make_outer_zip(archive, [PRODUCT_A], tmp_path)
    target = tmp_path / "scenes"
    zenodo.extract_scenes(archive, target)

    statuses = manifest.check_collection(target, [PRODUCT_A], min_bytes=1)
    assert statuses[0].usable


def test_download_raises_when_size_is_wrong(tmp_path: Path, monkeypatch) -> None:
    archive = zenodo.Archive("small.zip", expected_bytes=999)

    def fake_run(command, check=False):
        del command, check
        (tmp_path / "small.zip").write_bytes(b"short")

    def no_aria2() -> bool:
        return False

    monkeypatch.setattr(zenodo.subprocess, "run", fake_run)
    monkeypatch.setattr(zenodo, "_has_aria2", no_aria2)
    with pytest.raises(RuntimeError, match="expected 999"):
        zenodo.download(archive, tmp_path)


def test_download_skips_a_complete_file(tmp_path: Path, monkeypatch) -> None:
    archive = zenodo.Archive("small.zip", expected_bytes=4)
    (tmp_path / "small.zip").write_bytes(b"1234")

    def fail(command, check=False):
        del command, check
        raise AssertionError("download must not run for a complete file")

    monkeypatch.setattr(zenodo.subprocess, "run", fail)
    assert zenodo.download(archive, tmp_path).name == "small.zip"


def test_landsat_archives_point_at_their_own_record() -> None:
    assert zenodo.L8_RECORD_ID == "5040271"
    assert zenodo.L8_RECORD_ID in zenodo.L8_SCENES.url
    assert zenodo.L8_SCENES.url.endswith("Landsat8_L1.zip?download=1")
    assert zenodo.L8_LABELS.url.endswith("PixBox-L8-CMIX.zip?download=1")


def test_landsat_class_table_is_named_descriptions() -> None:
    """The Sentinel-2 collection calls the same file definitions."""
    assert zenodo.L8_DESCRIPTIONS.endswith("_descriptions.txt")
    assert zenodo.DEFINITIONS.endswith("_definitions.txt")


def test_landsat_archive_holds_every_declared_scene() -> None:
    from cloudband.labels.pixbox_l8 import PRODUCT_ID_TO_SCENE

    declared = len(PRODUCT_ID_TO_SCENE)
    assert declared == zenodo.L8_SCENES_IN_ARCHIVE