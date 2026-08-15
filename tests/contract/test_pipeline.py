"""Contract tests for the scoring pipeline and the inference wrapper."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cloudband.baselines.ocm import InferenceConfig, prediction_directory, validate_kwargs
from cloudband.eval.experiments import pool
from cloudband.eval.report import as_percentages, compare, to_frame
from cloudband.labels import pixbox
from cloudband.pipelines import pixbox_s2
from cloudband.provenance.manifest import build_manifest, file_digest

SCENE_SIZE = 8


def build_table(product_ids: list[int], rows: list[int], columns: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            pixbox.CLOUD_COLUMN: [6, 0, 0][: len(rows)],
            pixbox.SHADOW_COLUMN: [3, 3, 0][: len(rows)],
            pixbox.PRODUCT_COLUMN: product_ids,
            pixbox.ROW_COLUMN: rows,
            pixbox.COLUMN_COLUMN: columns,
        }
    )


def write_prediction(directory: Path, scene_name: str, array: np.ndarray) -> Path:
    rasterio = pytest.importorskip("rasterio")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{scene_name}_OCM_v1.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
    ) as destination:
        destination.write(array, 1)
    return path


def test_inference_config_serialises_pinned_options() -> None:
    kwargs = InferenceConfig().as_kwargs()
    assert kwargs["patch_size"] == 1000
    assert kwargs["patch_overlap"] == 300
    assert "inference_device" not in kwargs


def test_inference_config_includes_device_when_set() -> None:
    kwargs = InferenceConfig(inference_device="cuda").as_kwargs()
    assert kwargs["inference_device"] == "cuda"


def test_unsupported_keyword_is_rejected() -> None:
    def target(supported: int = 0) -> int:
        return supported

    with pytest.raises(TypeError, match="does not accept"):
        validate_kwargs(target, {"supported": 1, "invented": 2})


def test_keyword_passthrough_when_function_accepts_any() -> None:
    def target(**kwargs: int) -> int:
        return len(kwargs)

    assert validate_kwargs(target, {"anything": 1}) == {"anything": 1}


def test_prediction_directory_is_version_scoped() -> None:
    assert prediction_directory(Path("/tmp"), "1.7.0").name == "ocm_preds_v1.7.0"


def test_wrong_raster_size_is_rejected(tmp_path: Path) -> None:
    array = np.zeros((SCENE_SIZE, SCENE_SIZE), dtype=np.uint8)
    path = write_prediction(tmp_path, "scene", array)
    with pytest.raises(ValueError, match="label pixel indices assume"):
        pixbox_s2.read_prediction(path, expected_size=10980)


def test_duplicate_predictions_are_rejected(tmp_path: Path) -> None:
    array = np.zeros((SCENE_SIZE, SCENE_SIZE), dtype=np.uint8)
    write_prediction(tmp_path, "scene", array)
    (tmp_path / "scene_OCM_v2.tif").write_bytes(b"")
    with pytest.raises(ValueError, match="predictions match"):
        pixbox_s2.find_prediction(tmp_path, "scene")


def test_missing_prediction_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no prediction"):
        pixbox_s2.find_prediction(tmp_path, "absent")


def test_attach_and_score_round_trip(tmp_path: Path) -> None:
    product_id = next(iter(pixbox.PRODUCT_ID_TO_SCENE))
    scene_name = pixbox.PRODUCT_ID_TO_SCENE[product_id]
    prediction = np.zeros((SCENE_SIZE, SCENE_SIZE), dtype=np.uint8)
    prediction[1, 2] = 1
    prediction[3, 4] = 3
    write_prediction(tmp_path, scene_name, prediction)

    table = build_table([product_id] * 3, [1, 3, 0], [2, 4, 0])
    attached = pixbox_s2.attach_predictions(table, tmp_path, expected_size=SCENE_SIZE)
    assert attached[pixbox_s2.PREDICTION_COLUMN].tolist() == [1, 3, 0]

    confusions = pixbox_s2.score(attached)
    assert confusions["cloud"].tp == 1
    assert confusions["shadow"].tp == 1
    assert confusions["clear"].tp == 1


def test_unknown_product_id_is_rejected(tmp_path: Path) -> None:
    table = build_table([-1], [0], [0])
    with pytest.raises(ValueError, match="no scene name known"):
        pixbox_s2.attach_predictions(table, tmp_path, expected_size=SCENE_SIZE)


def test_score_requires_attached_predictions() -> None:
    table = build_table([next(iter(pixbox.PRODUCT_ID_TO_SCENE))], [0], [0])
    with pytest.raises(ValueError, match="call attach_predictions first"):
        pixbox_s2.score(table)


def test_pooled_counts_equal_whole_collection(tmp_path: Path) -> None:
    product_id = next(iter(pixbox.PRODUCT_ID_TO_SCENE))
    scene_name = pixbox.PRODUCT_ID_TO_SCENE[product_id]
    prediction = np.zeros((SCENE_SIZE, SCENE_SIZE), dtype=np.uint8)
    write_prediction(tmp_path, scene_name, prediction)
    table = build_table([product_id] * 3, [1, 3, 0], [2, 4, 0])
    attached = pixbox_s2.attach_predictions(table, tmp_path, expected_size=SCENE_SIZE)

    pooled = pool(pixbox_s2.score_per_scene(attached))
    whole = pixbox_s2.score(attached)
    for name in whole:
        assert pooled[name] == whole[name]


def test_report_columns_and_percentages() -> None:
    reference = {"clear": np.array([True, False]), "cloud": np.array([False, True])}
    reference["shadow"] = np.array([False, False])
    from cloudband.eval.experiments import run_experiments

    confusions = run_experiments(reference, reference)
    frame = to_frame(confusions)
    assert list(frame.index) == ["clear", "cloud", "shadow"]
    assert frame.loc["clear", "boa"] == 1.0
    assert as_percentages(frame).loc["clear", "boa"] == 100.0
    assert list(compare({"run": frame}).columns) == ["run"]


def test_manifest_records_environment_and_digests(tmp_path: Path) -> None:
    source = tmp_path / "input.csv"
    source.write_text("a,b\n1,2\n", encoding="utf-8")
    manifest = build_manifest(
        result_name="pixbox_s2_boa",
        dataset="pixbox_s2",
        model_id="ocm-rgn-published-v1.7.0",
        config={"patch_size": 1000},
        inputs={"labels": source},
    )
    written = manifest.write(tmp_path / "manifest.json")
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["model_id"] == "ocm-rgn-published-v1.7.0"
    assert payload["inputs"]["labels"] == file_digest(source)
    assert payload["config"]["patch_size"] == 1000


def test_session_requires_existing_project(tmp_path: Path) -> None:
    from cloudband.colab.session import start

    with pytest.raises(RuntimeError, match="clone the repository first"):
        start(tmp_path / "absent")


def test_session_creates_data_and_results(tmp_path: Path) -> None:
    from cloudband.colab.session import start

    session = start(tmp_path)
    assert session.data_dir.is_dir()
    assert session.results_dir.is_dir()
    assert "numpy" in session.package_versions


def test_package_versions_marks_absent() -> None:
    from cloudband.colab.session import package_versions

    assert package_versions(("definitely_not_installed",)) == {"definitely_not_installed": "absent"}


def test_reload_package_clears_only_its_modules() -> None:
    from cloudband.colab.session import reload_package

    sys_modules_before = set(__import__("sys").modules)
    assert any(name.startswith("cloudband") for name in sys_modules_before)
    reload_package("cloudband")
    remaining = {name for name in __import__("sys").modules if name.startswith("cloudband")}
    assert not remaining


def test_require_disk_rejects_impossible_threshold(tmp_path: Path) -> None:
    from cloudband.colab.session import require_disk

    with pytest.raises(RuntimeError, match="GiB required"):
        require_disk(minimum_gb=10.0**9, path=tmp_path)