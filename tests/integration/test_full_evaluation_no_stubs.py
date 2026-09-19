import numpy as np
import torch

from cloudband.datasets.cloudsen12 import NODATA_VALUE, Sample
from cloudband.pipelines.cloudsen12 import (
    pairable_scene_counts,
    per_scene_metric,
    pooled,
    score_split,
)
from cloudband.eval.metrics import balanced_overall_accuracy
from cloudband.train.comparison import compare_per_scene_boa
from cloudband.train.predictor import build_predictor

PATCH_SIZE = 32
STACK_CHANNELS = 13


class TinySegmenter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def realistic_annotation(seed: int) -> np.ndarray:
    """A dense annotation with all four classes, exclusive, no overlap."""
    rng = np.random.default_rng(seed)
    annotation = rng.integers(0, 4, size=(PATCH_SIZE, PATCH_SIZE)).astype(np.int32)
    annotation[0, :3] = NODATA_VALUE
    return annotation


def fake_samples():
    for index in range(3):
        rng = np.random.default_rng(index)
        image = rng.integers(0, 10000, size=(STACK_CHANNELS, PATCH_SIZE, PATCH_SIZE))
        yield Sample(
            identifier=f"scene-{index}",
            image=image.astype(np.int32),
            annotation=realistic_annotation(index),
        )


def test_score_split_runs_end_to_end_with_every_real_module():
    predictor = build_predictor(TinySegmenter())

    per_scene = score_split(fake_samples(), predictor)

    assert set(per_scene) == {"scene-0", "scene-1", "scene-2"}
    for confusions in per_scene.values():
        assert set(confusions) == {"clear", "cloud", "shadow"}


def test_score_split_excludes_nodata_pixels_from_every_confusion():
    predictor = build_predictor(TinySegmenter())
    per_scene = score_split(fake_samples(), predictor)

    for confusions in per_scene.values():
        total_counted = confusions["clear"].total
        annotated_pixels = PATCH_SIZE * PATCH_SIZE - 3
        assert total_counted == annotated_pixels


def test_pooled_and_per_scene_metric_and_comparison_all_agree():
    predictor_a = build_predictor(TinySegmenter())
    predictor_b = build_predictor(TinySegmenter())

    per_scene_a = score_split(fake_samples(), predictor_a)
    per_scene_b = score_split(fake_samples(), predictor_b)

    pooled_a = pooled(per_scene_a)
    assert set(pooled_a) == {"clear", "cloud", "shadow"}

    boa_a = per_scene_metric(per_scene_a, balanced_overall_accuracy)
    boa_b = per_scene_metric(per_scene_b, balanced_overall_accuracy)

    pairable = pairable_scene_counts(boa_a)
    assert all(count <= 3 for count in pairable.values())

    results = compare_per_scene_boa(boa_a, boa_b)
    assert {r.experiment for r in results} == {"clear", "cloud", "shadow"}