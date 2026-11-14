import types

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from cloudband.baselines import ocm
from cloudband.eval import phase2 as evaluation
from cloudband.eval.metrics import balanced_overall_accuracy
from cloudband.pipelines import phase0, phase2_eval
from cloudband.pipelines.cloudsen12 import per_scene_metric, score_split
from tests.unit.test_phase2_evaluation import annotation, models, samples

TRUTH = annotation(0)
TRUTH[TRUTH == 99] = 0
SPLITS = {"test": None, "test_l2a": None}
LEVELS = ("L1C", "L2A")


def published_prediction(version):
    """Each version gets a different share of pixels wrong; 4.0 is the truth."""
    prediction = TRUTH.copy()
    wrong = np.random.default_rng(int(version)).random(TRUTH.shape) < (4 - version) / 10
    prediction[wrong] = (prediction[wrong] + 1) % 4
    return prediction


class Store:
    def __init__(self, done=True):
        self.done = done

    def is_done(self, name):
        return self.done


@pytest.fixture
def setting(tmp_path, monkeypatch):
    data = samples(count=20)
    calls = []
    monkeypatch.setattr(ocm, "package_version", lambda: "1.7.0")
    monkeypatch.setattr(
        ocm, "predict_array",
        lambda stack, config: calls.append(config.model_version)
        or published_prediction(config.model_version),
    )
    cache = types.SimpleNamespace(samples=lambda split, table: iter(data))
    loads = []

    def load(seed):
        loads.append(seed)
        return models()

    return types.SimpleNamespace(
        data=data, cache=cache, load=load, loads=loads, calls=calls, results=tmp_path
    )


def run(setting, store=None):
    published = phase2_eval.score_published(setting.cache, SPLITS, setting.results, log=lambda m: None)
    counts = phase2_eval.score_trained(
        setting.cache, SPLITS, store or Store(), setting.results, "cpu",
        load=setting.load, log=lambda m: None,
    )
    combined = phase2_eval.with_published(counts, published)
    return published, counts, combined, phase2_eval.build_reports(combined)


def test_every_version_seed_and_level_is_scored_once_then_read_back(setting):
    published, counts, _, _ = run(setting)
    assert sorted(counts) == ["test", "test_l2a"]
    assert sorted(counts["test"]) == [0, 1, 2, 3, 4]
    assert setting.loads == [0, 1, 2, 3, 4]
    assert set(published["test"]["model"]) == set(phase2_eval.PUBLISHED)
    # one version over every patch before the next, so the package loads it once
    per_level = [v for v in phase2_eval.PUBLISHED_VERSIONS for _ in range(20)]
    assert setting.calls == per_level * 2

    run(setting)
    assert setting.loads == [0, 1, 2, 3, 4]
    assert len(setting.calls) == len(per_level) * 2   # the second run reads the saved counts


def test_counts_saved_under_an_older_model_name_are_renamed(setting):
    path = phase2_eval.counts_path(setting.results, "test", phase0.published_model_id(4.0))
    evaluation.save_counts(
        evaluation.score_predictor(setting.data, lambda image: TRUTH, "ocm published"), path
    )
    published, *_ = run(setting)
    assert "ocm published" not in set(published["test"]["model"])
    assert (published["test"]["model"] == phase2_eval.published_name(4.0)).sum() == 20 * 3


def test_nothing_is_reported_before_a_seed_has_all_its_networks(setting):
    _, counts, combined, reports = run(setting, Store(done=False))
    assert counts == {"test": {}, "test_l2a": {}}
    assert combined == {} and reports == {}


def test_each_level_has_its_own_report_with_every_published_version(setting):
    _, _, combined, reports = run(setting)
    assert tuple(reports) == LEVELS
    for name in phase2_eval.PUBLISHED:
        assert f"OCM vs {name}, per ROI" in reports["L1C"]
        assert f"Swin vs {name}, per ROI" in reports["L1C"]
    expected = evaluation.report(combined["test"], published=phase2_eval.PUBLISHED)
    pd.testing.assert_frame_equal(
        reports["L1C"]["OCM vs Swin, per ROI"], expected["OCM vs Swin, per ROI"]
    )
    assert phase2_eval.level_agreement(reports) == {"clear": True, "cloud": True, "shadow": True}


def test_the_tables_are_in_points_and_name_every_model(setting):
    _, _, _, reports = run(setting)
    everything = phase2_eval.table_all_models(reports, "L1C")
    assert list(everything.index.get_level_values(0).unique()) == [
        phase2_eval.LABELS[m] for m in (*phase2_eval.PUBLISHED, *phase2_eval.TRAINED)
    ]
    assert everything["mean"].max() > 1   # percent, not a fraction
    expected = evaluation.pooled_boa(run(setting)[0]["test"]).loc["ocm published 4.0", "cloud"]
    assert everything.loc[("OCM published 4.0", "cloud"), "mean"] == pytest.approx(expected * 100)

    swin = phase2_eval.table_ocm_vs_swin(reports, "L1C")
    raw = reports["L1C"]["OCM vs Swin, per ROI"]
    assert swin.loc[("OCM vs Swin", "cloud"), "mean diff"] == pytest.approx(
        raw.loc["cloud", "mean diff"] * 100
    )
    assert swin.loc[("OCM vs Swin", "cloud"), "p"] == raw.loc["cloud", "p"]

    versus = phase2_eval.table_vs_published(reports, "L1C")
    assert len(versus.index.get_level_values(0).unique()) == 2 * len(phase2_eval.PUBLISHED)


def test_published_versions_alone_give_whole_split_and_paper_tables(setting):
    published, *_ = run(setting)
    scores = phase2_eval.published_scores(published)
    pooled = evaluation.pooled_boa(published["test_l2a"]) * 100
    for version in phase2_eval.PUBLISHED_VERSIONS:
        name = phase2_eval.published_name(version)
        assert scores.loc[("L2A", phase2_eval.LABELS[name], "shadow"), "boa"] == pytest.approx(
            pooled.loc[name, "shadow"], abs=0.005
        )   # as_percentages rounds to two decimals
    paper = phase2_eval.published_cloudsen12_tables(published)
    assert set(paper.index.get_level_values(0)) == set(LEVELS)
    assert len(paper.index.get_level_values(1).unique()) == len(phase2_eval.PUBLISHED)


def test_every_figure_is_built_and_saved(setting, tmp_path):
    _, _, combined, reports = run(setting)
    shown = []
    for level in LEVELS:
        for name, figure in {
            "all": phase2_eval.figure_all_models(reports, level),
            "swin": phase2_eval.figure_ocm_vs_swin(reports, level),
            "wins": phase2_eval.figure_roi_wins(combined, level),
            "published": phase2_eval.figure_vs_published(reports, level),
            "roi": phase2_eval.figure_roi_vs_published(combined, level),
        }.items():
            assert isinstance(figure, Figure)
            assert [ax.get_title() for ax in figure.axes][:3] == ["clear", "cloud", "shadow"]
            phase2_eval.show(figure, tmp_path / "figures", f"{level}-{name}", shown.append)
    phase2_eval.show(phase2_eval.figure_levels(reports), tmp_path / "figures", "levels", shown.append)

    assert len(shown) == 11
    assert len(list((tmp_path / "figures").glob("*.png"))) == 11


def test_the_cache_and_the_remote_file_give_the_same_published_boa(setting, tmp_path):
    published, *_ = run(setting)
    for version in (1.0, 4.0):
        directory = (
            tmp_path / "baseline" / phase0.published_model_id(version) / phase0.CLOUDSEN12_TEST
        )
        directory.mkdir(parents=True)
        per_scene_metric(
            score_split(setting.data, lambda image, v=version: published_prediction(v)),
            balanced_overall_accuracy,
        ).to_csv(directory / "per_scene_boa.csv")

    gaps = phase2_eval.published_gaps(published, tmp_path / "baseline")
    assert list(gaps["model"]) == ["OCM published 1.0", "OCM published 4.0"]
    assert gaps["largest gap"].abs().max() == pytest.approx(0.0)
    assert phase2_eval.published_gaps(published, tmp_path / "absent").empty


def test_legacy_gaps_report_only_the_files_that_exist(setting):
    _, counts, _, _ = run(setting)
    assert phase2_eval.legacy_gaps(counts["test"], setting.results).empty

    frame = evaluation.unit_boa(counts["test"][0], "scene").loc["swin"]
    frame.to_csv(setting.results / "swin_shared_per_scene_boa_seed0.csv")
    gaps = phase2_eval.legacy_gaps(counts["test"], setting.results)
    assert gaps[["seed", "model"]].values.tolist() == [[0, "swin"]]
    assert gaps["largest gap"].iloc[0] == pytest.approx(0.0)


def test_the_fast_reader_check_refuses_a_mismatch(monkeypatch):
    table = pd.DataFrame({"a": range(3)})
    good = samples(count=1)[0]
    stack = np.zeros((13, *good.image.shape[1:]), dtype=good.image.dtype)
    stack[list(phase0.RGN_INDICES)] = good.image
    full = types.SimpleNamespace(image=stack, annotation=good.annotation)
    monkeypatch.setattr(phase2_eval, "gdal_read_sample", lambda table, row: full)

    assert phase2_eval.check_fast_reader(lambda t, row: good, {"test": table, "l2a": table}) == 3
    wrong = types.SimpleNamespace(image=good.image + 1, annotation=good.annotation)
    with pytest.raises(AssertionError, match="differs from GDAL"):
        phase2_eval.check_fast_reader(lambda t, row: wrong, {"test": table})
