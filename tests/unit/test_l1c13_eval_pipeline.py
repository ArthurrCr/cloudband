import types

import pandas as pd
import pytest
from matplotlib.figure import Figure

from cloudband.pipelines import l1c13_eval, phase2_eval
from tests.unit.test_phase2_eval_pipeline import (
    Store,
    setting,  # noqa: F401
)
from tests.unit.test_phase2_eval_pipeline import run as run_phase2
from tests.unit.test_phase2_evaluation import Fixed, annotation


def thirteen_band_models():
    """Other fakes than the R-G-NIR ones, so the two experiments differ."""
    truth = annotation(0)
    return {
        "regnety": Fixed(truth, strength=2.0, seed=4),
        "convnext": Fixed(truth, strength=2.5, seed=5),
        "swin": Fixed(truth, strength=1.5, seed=6),
    }


@pytest.fixture
def scored(setting):  # noqa: F811
    run_phase2(setting)
    loads = []

    def load(seed):
        loads.append(seed)
        return thirteen_band_models()

    counts = l1c13_eval.score_trained(
        setting.cache, None, Store(), setting.results, "cpu", load=load, log=lambda m: None
    )
    rgn = l1c13_eval.load_rgn_counts(setting.results, sorted(counts))
    published = l1c13_eval.load_published(setting.results)
    combined = l1c13_eval.combine(counts, rgn, published)
    return types.SimpleNamespace(
        setting=setting, counts=counts, loads=loads, combined=combined,
        report=l1c13_eval.build_report(combined),
    )


def test_the_13_band_seeds_are_scored_once_under_their_own_names(scored):
    assert scored.loads == [0, 1, 2, 3, 4]
    for seed in range(5):
        path = l1c13_eval.seed_counts_path(scored.setting.results, seed)
        assert path.name == f"phase2-v2-crop2-counts-test-l1c13-seed{seed}.csv"
        assert path.exists()
        assert path != phase2_eval.counts_path(scored.setting.results, "test", f"seed{seed}")

    again = l1c13_eval.score_trained(
        scored.setting.cache, None, Store(), scored.setting.results, "cpu",
        load=lambda seed: pytest.fail("a saved seed was scored again"), log=lambda m: None,
    )
    assert sorted(again) == [0, 1, 2, 3, 4]


def test_each_seed_holds_both_experiments_and_every_published_version(scored):
    expected = {*l1c13_eval.TRAINED, *l1c13_eval.RGN, *phase2_eval.PUBLISHED}
    for frame in scored.combined.values():
        assert set(frame["model"]) == expected


def test_the_checkpoints_are_those_of_the_13_band_run_ids():
    names = l1c13_eval.checkpoint_names(2)
    assert names["swin"] == "swin-l1c13-cs12-shared-v2-seed2"
    assert names["regnety"].startswith("ocm-l1c13-cs12-shared-v2-seed2-regnety")
    assert names["convnext"].startswith("ocm-l1c13-cs12-shared-v2-seed2-convnext")


def test_the_report_compares_the_13_band_models_and_the_bands(scored):
    report = scored.report
    for title in (*phase2_eval.VERSUS_SWIN, *l1c13_eval.BANDS_VS_RGN):
        assert set(report[title].index) == {"clear", "cloud", "shadow"}
    bands = report["OCM 13 bands vs OCM R-G-NIR, per ROI"]
    assert bands["mean diff"].to_numpy() == pytest.approx(
        (bands["boa ocm"] - bands["boa ocm rgn"]).to_numpy()
    )

    table = l1c13_eval.table_bands(report)
    assert set(table.index.get_level_values(0)) == {
        "OCM 13 bands vs OCM R-G-NIR", "Swin 13 bands vs Swin R-G-NIR",
    }
    in_points = table.loc["OCM 13 bands vs OCM R-G-NIR", "mean diff"].to_numpy()
    assert in_points == pytest.approx(bands["mean diff"].to_numpy() * 100)

    published = l1c13_eval.table_vs_published(report)
    assert "OCM 13 bands − published 4.0" in published.index.get_level_values(0)


def test_every_model_is_named_in_the_all_models_table(scored):
    table = l1c13_eval.table_all_models(scored.report)
    names = list(dict.fromkeys(table.index.get_level_values(0)))
    assert names[:4] == [f"OCM published {v:.1f}" for v in phase2_eval.PUBLISHED_VERSIONS]
    assert "OCM R-G-NIR (ensemble)" in names and "OCM 13 bands (ensemble)" in names


def test_the_figures_are_built(scored):
    assert isinstance(l1c13_eval.figure_all_models(scored.report), Figure)
    for name in ("clear", "cloud", "shadow"):
        assert isinstance(l1c13_eval.figure_by_share(scored.combined, name), Figure)
    share = l1c13_eval.table_by_share(scored.combined)
    assert list(share.columns) == ["patches", *[l1c13_eval.LABELS[m] for m in l1c13_eval.SHARE_MODELS]]


def test_the_report_is_saved_apart_from_the_phase2_one(scored, tmp_path):
    paths = l1c13_eval.save_report(scored.report, tmp_path)
    assert paths and all(p.name.startswith("phase2-v2-report-test-l1c13-") for p in paths)


def test_missing_phase2_counts_name_the_notebook_that_makes_them(tmp_path):
    with pytest.raises(FileNotFoundError, match="score_ocm_cloudsen12"):
        l1c13_eval.load_rgn_counts(tmp_path, [0])
    with pytest.raises(FileNotFoundError, match="published version"):
        l1c13_eval.load_published(tmp_path)


def test_seeds_must_match_between_the_experiments(scored):
    rgn = {seed: frame for seed, frame in scored.counts.items() if seed != 4}
    with pytest.raises(ValueError, match=r"\[4\]"):
        l1c13_eval.combine(scored.counts, rgn, pd.DataFrame())
