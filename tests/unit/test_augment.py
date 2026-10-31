import numpy as np
import pytest
import torch
from fastai.data.core import DataLoaders
from fastai.learner import Learner
from fastai.losses import CrossEntropyLossFlat

from cloudband.datasets.cloudsen12 import NODATA_VALUE, Sample
from cloudband.train.augment import (
    GeometricAugmentation,
    augment_sample,
    random_rectangle,
    scene_edge,
    tear,
)
from cloudband.train.data import CloudSen12Dataset, build_dataloaders
from cloudband.train.loop import MixedResolutionCallback, fit_protocol
from cloudband.train.protocol import (
    AugmentationConfig,
    TrainProtocol,
    ocm_native_protocol,
    ocm_shared_protocol,
    swin_shared_protocol,
)

CONFIG = AugmentationConfig()


def raw_image(size=64, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(500, 4000, size=(3, size, size)).astype(np.float32)


def test_the_rectangle_fills_one_value_in_some_channels_and_nothing_else():
    torch.manual_seed(0)
    image = raw_image()
    original = image.copy()

    random_rectangle(image, CONFIG)

    changed = image != original
    assert changed.any()
    touched = [c for c in range(3) if changed[c].any()]
    assert 1 <= len(touched) <= 3
    rows, cols = np.nonzero(changed.any(axis=0))
    region = image[touched[0], rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
    assert np.unique(region).size == 1
    assert 0 <= region.flat[0] <= CONFIG.rectangle_max_fill


def test_the_rectangle_covers_the_configured_share_of_the_image():
    areas = []
    for seed in range(200):
        torch.manual_seed(seed)
        image = np.full((1, 100, 100), -1.0, dtype=np.float32)
        random_rectangle(image, CONFIG)
        areas.append(np.count_nonzero(image != -1.0) / image.size)
    covered = np.array(areas)
    assert covered.max() <= CONFIG.rectangle_max_area + 0.05


def test_the_scene_edge_marks_one_side_of_a_line_as_no_data():
    torch.manual_seed(3)
    image = raw_image()

    scene_edge(image, NODATA_VALUE)

    nodata = image[0] == NODATA_VALUE
    assert (image[1] == NODATA_VALUE).sum() == nodata.sum()
    assert 0 < nodata.sum() < nodata.size


def test_the_sample_augmentations_follow_their_probabilities():
    torch.manual_seed(0)
    image = raw_image(size=32)
    runs = 2000
    rectangles = edges = 0
    for _ in range(runs):
        out = augment_sample(image, CONFIG, NODATA_VALUE)
        edges += bool((out == NODATA_VALUE).any())
        rectangles += bool(((out != image) & (out != NODATA_VALUE)).any())
    assert rectangles / runs == pytest.approx(CONFIG.random_rectangle_prob, abs=0.05)
    assert edges / runs == pytest.approx(CONFIG.scene_edge_prob, abs=0.03)


def test_the_sample_augmentations_never_touch_the_input():
    image = raw_image()
    original = image.copy()
    for seed in range(20):
        torch.manual_seed(seed)
        augment_sample(image, CONFIG, NODATA_VALUE)
    assert np.array_equal(image, original)


def test_the_same_seed_gives_the_same_augmentations():
    torch.manual_seed(7)
    first = augment_sample(raw_image(), CONFIG, NODATA_VALUE)
    torch.manual_seed(7)
    second = augment_sample(raw_image(), CONFIG, NODATA_VALUE)
    assert np.array_equal(first, second)


def test_the_tear_moves_image_and_label_together():
    label = torch.arange(64 * 64).reshape(1, 64, 64).repeat(2, 1, 1)
    image = label.unsqueeze(1).repeat(1, 3, 1, 1).float()

    torn_image = tear(image, 30.0, 32.0, 32.0, 10)
    torn_label = tear(label, 30.0, 32.0, 32.0, 10)

    assert torn_image.shape == image.shape
    assert torn_label.shape == label.shape
    assert torn_label.dtype == label.dtype
    assert torch.equal(torn_image[:, 0].long(), torn_label)
    assert not torch.equal(torn_label, label)


def augment_once(seed, training=True, config=CONFIG):
    torch.manual_seed(seed)
    label = torch.arange(64 * 64).reshape(1, 64, 64).repeat(2, 1, 1)
    image = label.unsqueeze(1).repeat(1, 3, 1, 1).float()
    callback = GeometricAugmentation(config)

    class FakeLearner:
        pass

    callback.learn = FakeLearner()
    callback.learn.xb, callback.learn.yb = (image,), (label,)
    callback.learn.training = training
    callback.before_batch()
    return callback, callback.learn.xb[0], callback.learn.yb[0], label


def test_the_geometric_augmentation_keeps_image_and_label_aligned():
    for seed in range(30):
        _, image, label, _ = augment_once(seed)
        for channel in range(3):
            assert torch.equal(image[:, channel].long(), label)


def test_rotations_and_flips_are_drawn_and_validation_is_left_alone():
    seen = {augment_once(seed)[0].last["rotation"] for seed in range(40)}
    assert seen == {0, 1, 2, 3}

    callback, image, label, original = augment_once(0, training=False)
    assert torch.equal(label, original)
    assert callback.last == {}


def test_a_tear_happens_on_about_a_tenth_of_the_batches():
    tears = sum(augment_once(seed)[0].last["tear"] for seed in range(1000))
    assert tears / 1000 == pytest.approx(CONFIG.image_tear_prob, abs=0.03)


def test_native_validation_keeps_validation_batches_at_full_size():
    callback = MixedResolutionCallback(9.0, 22.0, np.random.default_rng(0), native_validation=True)

    class FakeLearner:
        pass

    callback.learn = FakeLearner()
    callback.learn.training = False
    callback.learn.xb = (torch.rand(1, 3, 509, 509),)
    callback.learn.yb = (torch.zeros(1, 509, 509, dtype=torch.int64),)
    callback.before_batch()
    assert callback.learn.xb[0].shape[-1] == 509
    assert callback.gsd_history == []

    callback.learn.training = True
    callback.before_batch()
    assert callback.learn.xb[0].shape[-1] != 509
    assert len(callback.gsd_history) == 1


def test_revision_2_protocols_augment_and_validate_at_native_resolution():
    for protocol in (
        ocm_shared_protocol(1e-3),
        swin_shared_protocol(1e-3),
        ocm_native_protocol(),
    ):
        assert protocol.augment
        assert protocol.native_validation
        assert protocol.run_id.endswith("-v2")


def test_only_the_training_split_is_augmented():
    table = __import__("pandas").DataFrame({"tortilla:id": ["a"]})

    def read(_table, _index, _crop=True):
        return Sample("a", raw_image(16).astype(np.int16), np.zeros((16, 16), np.int16))

    dls = build_dataloaders(table, table, 1, read_sample=read, bands_selected=True,
                            augmentation=CONFIG)
    assert dls.train.dataset.augmentation == CONFIG
    assert dls.valid.dataset.augmentation is None


class TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def tiny_dls(augmentation):
    table = __import__("pandas").DataFrame({"tortilla:id": [f"p{i}" for i in range(4)]})

    def read(_table, index, _crop=True):
        image = raw_image(32, seed=index).astype(np.int16)
        return Sample(f"p{index}", image, np.full((32, 32), index % 4, np.int16))

    return build_dataloaders(table, table, 2, read_sample=read, bands_selected=True,
                             augmentation=augmentation)


def tiny_protocol():
    return TrainProtocol(
        run_id="tiny-v2", architecture="ocm", learning_rate=1e-3,
        frozen_epochs=1, unfrozen_epochs=1, effective_batch_size=2,
        min_gsd_m=9.0, max_gsd_m=22.0, augment=True, native_validation=True,
    )


def test_fit_refuses_dataloaders_built_without_the_augmentations():
    learner = Learner(tiny_dls(None), TinyNet(), loss_func=CrossEntropyLossFlat(axis=1))
    with pytest.raises(ValueError, match="augmentation"):
        fit_protocol(learner, tiny_protocol(), seed=0)


def test_an_augmented_fit_runs_end_to_end(tmp_path):
    learner = Learner(tiny_dls(CONFIG), TinyNet(), loss_func=CrossEntropyLossFlat(axis=1),
                      path=tmp_path)
    result = fit_protocol(learner, tiny_protocol(), seed=0)
    assert len(result.valid_losses) == 2
    assert all(np.isfinite(result.valid_losses))


def test_an_image_too_small_for_the_tear_is_left_as_it_is():
    image = torch.rand(1, 3, 16, 16)
    assert torch.equal(tear(image, 30.0, 8.0, 8.0, 10), image)
