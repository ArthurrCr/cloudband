"""Contract tests for the baseline run orchestration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cloudband.datasets import cloudsen12 as dataset
from cloudband.pipelines import phase0


def annotation(values: list[int], size: int = 2) -> np.ndarray:
    return np.array(values, dtype=np.int16).reshape(size, size)


def band_stack(values: list[int], channels: int = 13) -> np.ndarray:
    base = annotation(values)
    return np.stack([base * (index + 1) for index in range(channels)])


def make_table(count: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            dataset.ID_COLUMN: [f"scene_{index}" for index in range(count)],
            dataset.ROI_COLUMN: [f"roi_{index}" for index in range(count)],
        }
    )


def fake_iterator(samples: list[dataset.Sample]):
    """Return a stand-in for iter_samples that ignores its arguments."""

    def iterate(table, limit=None):
        del table, limit
        return iter(samples)

    return iterate


def constant_predictor(values: list[int]):
    fixed = annotation(values)

    def predict(stack: np.ndarray) -> np.ndarray:
        del stack
        return fixed

    return predict


def test_band_indices_match_the_published_table() -> None:
    """B04 red, B03 green and B8A near-infrared in the L1C stack, zero-based."""
    assert phase0.RED_INDEX == 3
    assert phase0.GREEN_INDEX == 2
    assert phase0.NIR_INDEX == 8
    assert phase0.RGN_BANDS == ("B04", "B03", "B8A")


def test_select_rgn_takes_three_channels_in_order() -> None:
    stack = band_stack([1, 1, 1, 1])
    selected = phase0.select_rgn(stack)
    assert selected.shape[0] == 3
    assert np.array_equal(selected[0], stack[phase0.RED_INDEX])
    assert np.array_equal(selected[1], stack[phase0.GREEN_INDEX])
    assert np.array_equal(selected[2], stack[phase0.NIR_INDEX])


def test_select_rgn_rejects_short_stack() -> None:
    with pytest.raises(ValueError, match="need at least"):
        phase0.select_rgn(band_stack([1, 1, 1, 1], channels=4))


def test_predictor_receives_only_three_channels() -> None:
    seen: list[tuple[int, ...]] = []

    def predict(stack: np.ndarray) -> np.ndarray:
        seen.append(stack.shape)
        return annotation([0, 0, 0, 0])

    predictor = phase0.build_predictor(predict)
    predictor(band_stack([1, 1, 1, 1]))
    assert seen == [(3, 2, 2)]


def test_run_assembles_scores_and_pairable_counts(monkeypatch) -> None:
    samples = [
        dataset.Sample(f"scene_{index}", band_stack([1, 1, 1, 1]), annotation([0, 1, 2, 3]))
        for index in range(3)
    ]
    monkeypatch.setattr(dataset, "iter_samples", fake_iterator(samples))

    result = phase0.run(make_table(3), constant_predictor([0, 1, 1, 3]), "ocm-test")
    assert len(result.per_scene) == 3
    assert set(result.scores.index) == {"clear", "cloud", "shadow"}
    assert result.pooled["cloud"].tp == 6
    assert result.pairable == {"clear": 3, "cloud": 3, "shadow": 3}


def test_run_reports_progress(monkeypatch) -> None:
    samples = [
        dataset.Sample(f"scene_{index}", band_stack([1, 1, 1, 1]), annotation([0, 1, 2, 3]))
        for index in range(2)
    ]
    monkeypatch.setattr(dataset, "iter_samples", fake_iterator(samples))
    seen: list[tuple[int, int]] = []

    phase0.run(
        make_table(2),
        constant_predictor([0, 1, 1, 3]),
        "ocm-test",
        progress=lambda position, total: seen.append((position, total)),
    )
    assert seen == [(1, 2), (2, 2)]


def test_save_writes_every_artefact(monkeypatch, tmp_path: Path) -> None:
    samples = [dataset.Sample("scene_0", band_stack([1, 1, 1, 1]), annotation([0, 1, 2, 3]))]
    monkeypatch.setattr(dataset, "iter_samples", fake_iterator(samples))

    result = phase0.run(make_table(1), constant_predictor([0, 1, 1, 3]), "ocm-test")
    paths = phase0.save(result, tmp_path, config={"patch_size": 1000})

    assert set(paths) == {"scores", "confusion", "per_scene_boa", "manifest"}
    assert all(path.is_file() for path in paths.values())

    payload = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert payload["model_id"] == "ocm-test"
    assert payload["config"]["bands"] == ["B04", "B03", "B8A"]
    assert payload["config"]["scenes"] == 1
    assert payload["config"]["patch_size"] == 1000


def test_saved_confusion_matches_pooled(monkeypatch, tmp_path: Path) -> None:
    samples = [dataset.Sample("scene_0", band_stack([1, 1, 1, 1]), annotation([0, 1, 2, 3]))]
    monkeypatch.setattr(dataset, "iter_samples", fake_iterator(samples))

    result = phase0.run(make_table(1), constant_predictor([0, 1, 1, 3]), "ocm-test")
    paths = phase0.save(result, tmp_path)
    counts = pd.read_csv(paths["confusion"], index_col=0)
    assert counts.loc["cloud", "tp"] == result.pooled["cloud"].tp


def test_reference_requires_a_dataset() -> None:
    with pytest.raises(ValueError, match="must name the dataset"):
        phase0.Reference(name="somewhere", dataset_id="", boa={"cloud": 90.0})


def test_reference_rejects_unknown_experiments() -> None:
    with pytest.raises(ValueError, match="unknown experiments"):
        phase0.Reference(name="r", dataset_id="d", boa={"haze": 90.0})


def test_published_references_name_their_datasets() -> None:
    published = (
        phase0.PIXBOX_S2_PAPER,
        phase0.PIXBOX_S2_V1_7_0,
        phase0.PIXBOX_L8_PAPER,
        phase0.STUPMASK_CLOUDSEN12,
    )
    assert all(reference.dataset_id for reference in published)
    assert phase0.PIXBOX_S2_PAPER.dataset_id == phase0.PIXBOX_S2
    assert phase0.STUPMASK_CLOUDSEN12.dataset_id == phase0.CLOUDSEN12_TEST
    assert set(phase0.STUPMASK_CLOUDSEN12.boa) == {"cloud"}


def test_same_dataset_comparison_is_a_fidelity_check(monkeypatch) -> None:
    samples = [dataset.Sample("scene_0", band_stack([1, 1, 1, 1]), annotation([0, 1, 2, 3]))]
    monkeypatch.setattr(dataset, "iter_samples", fake_iterator(samples))

    result = phase0.run(make_table(1), constant_predictor([0, 1, 1, 3]), "ocm-test")
    reference = phase0.Reference(
        name="same place", dataset_id=result.dataset_id, boa={"clear": 90.0}
    )
    frame = phase0.compare_to_reference(result, reference)
    assert list(frame.columns) == ["observed", "reference", "difference", "same_dataset"]
    assert bool(frame["same_dataset"].all())
    assert "fidelity check" in phase0.describe_comparison(frame)
    assert frame.loc["clear", "difference"] == pytest.approx(frame.loc["clear", "observed"] - 90.0)


def test_cross_dataset_comparison_is_flagged(monkeypatch) -> None:
    """A PixBox reference against a CloudSEN12+ run is not a fidelity check."""
    samples = [dataset.Sample("scene_0", band_stack([1, 1, 1, 1]), annotation([0, 1, 2, 3]))]
    monkeypatch.setattr(dataset, "iter_samples", fake_iterator(samples))

    result = phase0.run(make_table(1), constant_predictor([0, 1, 1, 3]), "ocm-test")
    frame = phase0.compare_to_reference(result, phase0.PIXBOX_S2_V1_7_0)
    assert not bool(frame["same_dataset"].any())
    description = phase0.describe_comparison(frame)
    assert "cross-dataset" in description
    assert "does not measure reproduction fidelity" in description