import types

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from cloudband.baselines import ocm
from cloudband.eval import phase2 as evaluation
from cloudband.pipelines import phase0, phase2_eval
from cloudband.pipelines.cloudsen12 import per_scene_metric, score_split
from cloudband.eval.metrics import balanced_overall_accuracy

from tests.unit.test_phase2_evaluation import annotation, models, samples

TRUTH = annotation(0)
TRUTH[TRUTH == 99] = 0
SPLITS = {"test": None, "test_l2a": None}


class Store:
    def __init__(self, done=True):
        self.done = done

    def is_done(self, name):
        return self.done


@pytest.fixture
def setting(tmp_path, monkeypatch):
    data = samples(count=20)
    monkeypatch.setattr(ocm, "package_version", lambda: "1.7.0")
    monkeypatch.setattr(ocm, "predict_array", lambda stack, config: TRUTH)
    cache = types.SimpleNamespace(samples=lambda split, table: iter(data))
    loads = []

    def load(seed):
        loads.append(seed)
        return models()

    return types.SimpleNamespace(data=data, cache=cache, load=load, loads=loads, results=tmp_path)


def run(setting, store=None):
    published = phase2_eval.score_published(setting.cache, SPLITS, setting.results, log=lambda m: None)
    counts = phase2_eval.score_trained(
        setting.cache, SPLITS, store or Store(), setting.results, "cpu",
        load=setting.load, log=lambda m: None,
    )
    combined = phase2_eval.with_published(counts, published)
    return published, counts, combined, phase2_eval.build_reports(combined)


def test_every_seed_and_level_is_scored_once_then_read_back(setting):
    published, counts, _, _ = run(setting)
    assert sorted(counts) == ["test", "test_l2a"]
    assert sorted(counts["test"]) == [0, 1, 2, 3, 4]
    assert setting.loads == [0, 1, 2, 3, 4]
    assert set(published["test"]["model"]) == {phase2_eval.PUBLISHED}

    run(setting)
    assert setting.loads == [0, 1, 2, 3, 4]   # the second run reads the saved counts


def test_nothing_is_scored_before_a_seed_has_all_its_networks(setting):
    _, counts, combined, reports = run(setting, Store(done=False))
    assert counts == {"test": {}, "test_l2a": {}}
    assert combined == {} and reports == {}
    assert phase2_eval.build_figures(reports, combined) == {}


def test_each_level_has_its_own_report_and_the_l2a_one_is_never_pooled(setting):
    _, counts, combined, reports = run(setting)
    assert list(reports) == ["L1C", "L2A"]
    for title in phase2_eval.VERSUS_PUBLISHED + phase2_eval.VERSUS_SWIN:
        assert title in reports["L1C"]
    expected = evaluation.report(combined["test"], published=phase2_eval.PUBLISHED)
    pd.testing.assert_frame_equal(
        reports["L1C"]["OCM vs Swin, per ROI"], expected["OCM vs Swin, per ROI"]
    )
    assert phase2_eval.level_agreement(reports) == {"clear": True, "cloud": True, "shadow": True}


def test_reports_and_figures_are_saved(setting, tmp_path):
    _, _, combined, reports = run(setting)
    tables = phase2_eval.save_reports(reports, setting.results)
    figures = phase2_eval.build_figures(reports, combined)
    images = phase2_eval.save_figures(figures, tmp_path / "figures")

    assert len(tables) == 2 * len(reports["L1C"])
    assert all(isinstance(figure, Figure) for figure in figures.values())
    assert {"mean-vs-published", "differences-vs-published-per-roi",
            "roi-vs-published-test", "roi-vs-published-test_l2a"} <= set(figures)
    assert all(path.exists() for path in images)


def test_the_published_scores_from_the_cache_match_the_phase0_run(setting, tmp_path):
    published, *_ = run(setting)
    directory = tmp_path / "baseline" / phase0.published_model_id(4.0) / phase0.CLOUDSEN12_TEST
    directory.mkdir(parents=True)
    per_scene_metric(
        score_split(setting.data, lambda image: TRUTH), balanced_overall_accuracy
    ).to_csv(directory / "per_scene_boa.csv")

    assert phase2_eval.published_gap(published, tmp_path / "baseline") == pytest.approx(0.0)
    assert phase2_eval.published_gap(published, tmp_path / "absent") is None


def test_legacy_gaps_report_only_the_files_that_exist(setting):
    _, counts, _, _ = run(setting)
    assert phase2_eval.legacy_gaps(counts["test"], setting.results).empty

    frame = evaluation.unit_boa(counts["test"][0], "scene").loc["swin"]
    frame.to_csv(setting.results / "swin_shared_per_scene_boa_seed0.csv")
    gaps = phase2_eval.legacy_gaps(counts["test"], setting.results)
    assert gaps[["seed", "model"]].values.tolist() == [[0, "swin"]]
    assert gaps["largest gap"].iloc[0] == pytest.approx(0.0)


def test_published_versions_are_read_back_instead_of_run_again(tmp_path, monkeypatch):
    monkeypatch.setattr(ocm, "package_version", lambda: "1.7.0")
    directory = tmp_path / phase0.published_model_id(4.0) / phase0.CLOUDSEN12_TEST
    directory.mkdir(parents=True)
    saved = pd.DataFrame({"boa": [90.0, 91.0, 80.0]},
                         index=pd.Index(["clear", "cloud", "shadow"], name="experiment"))
    saved.to_csv(directory / "scores.csv")
    saved.to_csv(directory / "paper_table.csv")

    def never(*args, **kwargs):
        raise AssertionError("a saved version must not run again")

    monkeypatch.setattr(phase0, "run", never)
    scores, paper = phase0.score_published_versions(None, (4.0,), tmp_path, log=lambda m: None)
    pd.testing.assert_frame_equal(scores[4.0], saved)
    assert phase0.version_label(4.0) == "V4"


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
