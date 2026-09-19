import numpy as np

from cloudband.eval.confusion import confusion_from_masks
from cloudband.eval.experiments import EXPERIMENTS
from cloudband.eval.metrics import balanced_overall_accuracy
from cloudband.pipelines.cloudsen12 import pairable_scene_counts, per_scene_metric
from cloudband.train.comparison import compare_per_scene_boa


def confusions_for_scene(reference: dict, prediction: dict) -> dict:
    return {
        name: confusion_from_masks(name, reference[name], prediction[name])
        for name in EXPERIMENTS
    }


def perfect_masks(shape) -> dict:
    clear = np.zeros(shape, dtype=bool)
    clear[:4, :4] = True
    shadow = np.zeros(shape, dtype=bool)
    shadow[6:, 6:] = True
    cloud = ~clear & ~shadow
    return {"clear": clear, "cloud": cloud, "shadow": shadow}


def test_real_metrics_and_comparison_stack_end_to_end():
    shape = (10, 10)
    masks_a = perfect_masks(shape)

    masks_b = {name: mask.copy() for name, mask in masks_a.items()}
    masks_b["clear"][0, 0] = not masks_b["clear"][0, 0]
    masks_b["cloud"][0, 0] = not masks_b["cloud"][0, 0]

    per_scene_model_a = {
        "scene-1": confusions_for_scene(masks_a, masks_a),
        "scene-2": confusions_for_scene(masks_a, masks_a),
    }
    per_scene_model_b = {
        "scene-1": confusions_for_scene(masks_a, masks_b),
        "scene-2": confusions_for_scene(masks_a, masks_a),
    }

    boa_a = per_scene_metric(per_scene_model_a, balanced_overall_accuracy)
    boa_b = per_scene_metric(per_scene_model_b, balanced_overall_accuracy)

    assert boa_a.loc["scene-1", "clear"] == 1.0
    assert boa_b.loc["scene-1", "clear"] < 1.0
    assert boa_a.loc["scene-2", "clear"] == boa_b.loc["scene-2", "clear"] == 1.0

    pairable = pairable_scene_counts(boa_a)
    assert pairable == {"clear": 2, "cloud": 2, "shadow": 2}

    results = compare_per_scene_boa(boa_a, boa_b)
    clear_result = next(r for r in results if r.experiment == "clear")
    assert clear_result.n_pairs == 2
    assert clear_result.median_difference > 0


def test_a_scene_missing_a_class_entirely_drops_from_pairable_counts():
    shape = (4, 4)
    all_clear = {
        "clear": np.ones(shape, dtype=bool),
        "cloud": np.zeros(shape, dtype=bool),
        "shadow": np.zeros(shape, dtype=bool),
    }
    per_scene = {"all-clear-scene": confusions_for_scene(all_clear, all_clear)}

    boa = per_scene_metric(per_scene, balanced_overall_accuracy)

    assert np.isnan(boa.loc["all-clear-scene", "clear"])
    assert pairable_scene_counts(boa) == {"clear": 0, "cloud": 0, "shadow": 0}