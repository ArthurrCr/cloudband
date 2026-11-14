import math

import numpy as np
import pandas as pd
import pytest
import torch

from cloudband.datasets.cloudsen12 import Sample
from cloudband.eval import phase2 as evaluation
from cloudband.eval.metrics import balanced_overall_accuracy
from cloudband.models.ocm import OcmEnsemble
from cloudband.pipelines.cloudsen12 import per_scene_metric, score_split
from cloudband.train.predictor import build_predictor

SIZE = 12


class Fixed(torch.nn.Module):
    """Logits that favour one class map, softly enough for the ensemble to matter."""

    def __init__(self, classes, strength=2.0, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        logits = rng.normal(0, 1, (1, 4, SIZE, SIZE)).astype(np.float32)
        for c in range(4):
            logits[0, c][classes == c] += strength
        self.register_buffer("logits", torch.from_numpy(logits))

    def forward(self, x):
        return self.logits.expand(x.shape[0], -1, -1, -1)


def annotation(seed):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 4, (SIZE, SIZE))
    labels[:2, :2] = 99
    return labels


def samples(count=10, per_roi=5):
    rng = np.random.default_rng(42)
    return [
        Sample(
            identifier=f"p{i:03d}",
            image=rng.integers(100, 5000, (3, SIZE, SIZE)).astype(np.int16),
            annotation=annotation(i),
            roi_id=f"0{i // per_roi}",
        )
        for i in range(count)
    ]


def models(truth_seed=0):
    truth = annotation(truth_seed)
    return {
        "regnety": Fixed(truth, strength=1.0, seed=1),
        "convnext": Fixed(truth, strength=1.5, seed=2),
        "swin": Fixed(truth, strength=0.5, seed=3),
    }


def score(data=None, members=None):
    return evaluation.score_models(
        data or samples(), members or models(), {"ocm": ("regnety", "convnext")}
    )


def test_counts_cover_every_annotated_pixel_once_per_experiment():
    counts = score()

    assert set(counts["model"]) == {"regnety", "convnext", "ocm", "swin"}
    totals = counts.groupby(["model", "scene", "experiment"])[["tp", "tn", "fp", "fn"]].sum()
    assert (totals.sum(axis=1) == SIZE * SIZE - 4).all()
    assert counts.groupby("scene")["roi"].nunique().max() == 1


def test_the_per_patch_boa_matches_the_existing_scoring_pipeline():
    """The new single pass must give what build_predictor + score_split gave before."""
    data, members = samples(), models()
    counts = score(data, members)

    for name, model in (
        ("ocm", OcmEnsemble((members["regnety"], members["convnext"]))),
        ("swin", members["swin"]),
        ("regnety", members["regnety"]),
    ):
        old = per_scene_metric(
            score_split(data, build_predictor(model, "cpu", bands_selected=True)),
            balanced_overall_accuracy,
        )
        assert evaluation.legacy_check(counts, name, old) == pytest.approx(0.0, abs=1e-12)


def test_roi_counts_are_summed_before_the_boa_is_taken():
    counts = score()
    per_scene = counts[(counts.model == "swin") & (counts.experiment == "cloud")]
    first_roi = per_scene[per_scene.roi == "00"][["tp", "tn", "fp", "fn"]].sum()
    expected = 0.5 * (
        first_roi.tp / (first_roi.tp + first_roi.fn) + first_roi.tn / (first_roi.tn + first_roi.fp)
    )

    roi = evaluation.unit_boa(counts, "roi").loc["swin"]

    assert roi.loc["00", "cloud"] == pytest.approx(expected)
    assert len(roi) == 2


def test_a_class_absent_from_a_unit_leaves_its_boa_undefined():
    frame = pd.DataFrame(
        [("m", "s", "r", "cloud", 0, 10, 2, 0)], columns=list(evaluation.COLUMNS)
    )
    assert math.isnan(evaluation.boa(frame).iloc[0])


def test_holm_adjustment():
    adjusted = evaluation.holm([0.01, 0.04, 0.03, np.nan])
    assert adjusted[:3] == pytest.approx([0.03, 0.06, 0.06])
    assert math.isnan(adjusted[3])


def test_a_better_model_shows_a_positive_difference_with_an_interval_above_zero():
    data = samples(count=40)
    counts_by_seed = {}
    for seed in range(2):
        per_sample = []
        for sample in data:
            truth = sample.annotation.copy()
            truth[truth == 99] = 0
            per_sample.append(
                evaluation.score_models(
                    [sample],
                    {"regnety": Fixed(truth, 6.0, seed), "convnext": Fixed(truth, 6.0, seed + 5),
                     "swin": Fixed(truth, 0.3, seed + 9)},
                    {"ocm": ("regnety", "convnext")},
                )
            )
        counts_by_seed[seed] = pd.concat(per_sample, ignore_index=True)

    tables = evaluation.report(counts_by_seed)

    for name in ("OCM vs Swin, per ROI", "OCM vs Swin, per patch"):
        table = tables[name]
        assert (table["mean diff"] > 0).all()
        assert (table["ci95 low"] > 0).all()
        assert (table["p holm"] >= table["p"]).all()
    assert tables["OCM vs Swin, per ROI"]["n"].max() == 8
    assert tables["OCM vs Swin, per patch"]["n"].max() == 40
    seeds = tables["whole split, per seed"]
    assert list(seeds.columns[:2]) == ["mean", "sd"]
    assert seeds.loc[("ocm", "cloud"), "mean"] > seeds.loc[("swin", "cloud"), "mean"]


def test_counts_survive_a_round_trip_with_ids_that_look_like_numbers(tmp_path):
    counts = score()
    path = tmp_path / "counts.csv"

    evaluation.save_counts(counts, path)
    back = evaluation.load_counts(path)

    assert not (tmp_path / "counts.csv.partial").exists()
    assert set(back["roi"]) == {"00", "01"}
    pd.testing.assert_frame_equal(
        evaluation.unit_boa(back, "roi"), evaluation.unit_boa(counts, "roi")
    )


def test_an_ensemble_of_unknown_models_is_refused():
    with pytest.raises(ValueError, match="needs models"):
        evaluation.score_models(samples(), models(), {"ocm": ("regnety", "missing")})


def test_a_function_predictor_gives_the_counts_of_the_same_model():
    """The published model is reached through a function; same model, same counts."""
    from cloudband.train.normalize import dynamic_z_score

    data, members = samples(), models()
    swin = members["swin"]

    def predict(image):
        tensor = torch.from_numpy(dynamic_z_score(image.astype(np.float32))).unsqueeze(0)
        with torch.no_grad():
            return swin(tensor).argmax(dim=1).numpy()

    through_function = evaluation.score_predictor(data, predict, "swin")
    through_module = score(data, members).query("model == 'swin'").reset_index(drop=True)

    pd.testing.assert_frame_equal(through_function, through_module)


def test_the_published_model_joins_every_seed_and_gets_its_own_comparisons():
    data = samples()
    truth = annotation(0)
    truth[truth == 99] = 0
    published = evaluation.score_predictor(
        data, lambda image: truth, "ocm published"
    )
    counts_by_seed = evaluation.with_reference({0: score(data), 1: score(data)}, published)

    tables = evaluation.report(counts_by_seed, published="ocm published")

    assert "OCM vs ocm published, per ROI" in tables
    assert "Swin vs ocm published, per ROI" in tables
    assert tables["whole split, per seed"].loc[("ocm published", "cloud"), "sd"] == 0.0
    with pytest.raises(ValueError, match="different patches"):
        evaluation.with_reference({0: score(data)}, published.iloc[:3])
    with pytest.raises(ValueError, match="already has"):
        evaluation.with_reference(counts_by_seed, published)


def test_unit_differences_give_the_mean_difference_of_the_paired_table():
    counts_by_seed = {0: score(), 1: score(samples(), models(truth_seed=1))}
    differences = evaluation.unit_differences(counts_by_seed, "ocm", "swin", "roi")
    table = evaluation.paired_comparison(counts_by_seed, "ocm", "swin", "roi")

    for experiment in ("clear", "cloud", "shadow"):
        assert differences[experiment].dropna().mean() == pytest.approx(
            table.loc[experiment, "mean diff"]
        )
