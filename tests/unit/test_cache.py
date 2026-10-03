import numpy as np
import pandas as pd
import pytest
import torch

from cloudband.datasets.cloudsen12 import Sample
from cloudband.pipelines.phase0 import select_rgn
from cloudband.train import cache as cache_module
from cloudband.train.cache import LocalCache, subset_reader, table_fingerprint
from cloudband.train.data import CloudSen12Dataset
from cloudband.train.predictor import build_predictor

PATCH = 24


def fake_read_sample(table, index, crop_to_valid=True):
    rng = np.random.default_rng(index)
    return Sample(
        identifier=f"scene-{index:03d}",
        image=rng.integers(0, 10000, (13, PATCH, PATCH)).astype(np.uint16),
        annotation=rng.integers(0, 4, (PATCH, PATCH)).astype(np.uint8),
        roi_id=f"roi-{index % 3}",
    )


def make_table(n=6):
    return pd.DataFrame({"tortilla:id": [f"id{i}" for i in range(n)]})


def quiet(_message):
    return None


def build(tmp_path, table=None, **kwargs):
    table = make_table() if table is None else table
    cache = LocalCache(tmp_path / "cache")
    cache.build("train", table, read_sample=fake_read_sample, progress=quiet,
                workers=2, backoff_seconds=0, pause_seconds=0, **kwargs)
    return cache, table


def test_a_cached_sample_keeps_only_the_three_bands_and_everything_else(tmp_path):
    cache, table = build(tmp_path)
    cached = cache.reader("train", table)(table, 2)
    original = fake_read_sample(table, 2)

    assert cached.image.shape == (3, PATCH, PATCH)
    assert np.array_equal(cached.image, select_rgn(original.image))
    assert cached.image.dtype == original.image.dtype
    assert np.array_equal(cached.annotation, original.annotation)
    assert cached.annotation.dtype == original.annotation.dtype
    assert cached.identifier == original.identifier
    assert cached.roi_id == original.roi_id


def test_the_dataset_returns_identical_tensors_from_the_cache(tmp_path):
    cache, table = build(tmp_path)
    reader = cache.reader("train", table)

    from_source = CloudSen12Dataset(table, read_sample=fake_read_sample)
    from_cache = CloudSen12Dataset(table, read_sample=reader, bands_selected=True)

    for index in range(len(table)):
        image_a, annotation_a = from_source[index]
        image_b, annotation_b = from_cache[index]
        assert torch.equal(image_a, image_b)
        assert torch.equal(annotation_a, annotation_b)


def test_the_predictor_gives_the_same_classes_on_a_cached_stack(tmp_path):
    torch.manual_seed(0)
    model = torch.nn.Conv2d(3, 4, kernel_size=1)
    full = fake_read_sample(None, 1).image

    on_full = build_predictor(model, device="cpu")(full)
    on_selected = build_predictor(model, device="cpu", bands_selected=True)(
        select_rgn(full)
    )

    assert np.array_equal(on_full, on_selected)


def test_building_again_only_copies_what_is_missing(tmp_path):
    cache, table = build(tmp_path)
    directory = cache.split_dir("train")
    (directory / "000001.npz").unlink()
    (directory / "000004.npz").unlink()
    calls = []

    def counting(table_, index, crop_to_valid=True):
        calls.append(index)
        return fake_read_sample(table_, index, crop_to_valid)

    cache.build("train", table, read_sample=counting, progress=quiet, workers=2)

    assert sorted(calls) == [1, 4]
    assert cache.reader("train", table) is not None


def test_a_failing_sample_is_reported_and_the_rest_are_kept(tmp_path):
    table = make_table()
    cache = LocalCache(tmp_path / "cache")

    def broken_on_three(table_, index, crop_to_valid=True):
        if index == 3:
            raise OSError("network")
        return fake_read_sample(table_, index, crop_to_valid)

    with pytest.raises(RuntimeError, match="1 samples failed"):
        cache.build("train", table, read_sample=broken_on_three, progress=quiet,
                    workers=2, retries=1, backoff_seconds=0, pause_seconds=0)

    assert len(list(cache.split_dir("train").glob("*.npz"))) == 5
    with pytest.raises(RuntimeError, match="5 of 6"):
        cache.reader("train", table)

    cache.build("train", table, read_sample=fake_read_sample, progress=quiet, workers=2)
    assert cache.reader("train", table) is not None


def test_a_transient_failure_is_retried(tmp_path):
    table = make_table(3)
    cache = LocalCache(tmp_path / "cache")
    attempts = {"n": 0}

    def flaky(table_, index, crop_to_valid=True):
        if index == 1 and attempts["n"] < 2:
            attempts["n"] += 1
            raise OSError("timeout")
        return fake_read_sample(table_, index, crop_to_valid)

    cache.build("train", table, read_sample=flaky, progress=quiet, workers=1,
                retries=3, backoff_seconds=0)

    assert attempts["n"] == 2
    assert cache.reader("train", table) is not None


def test_a_table_in_another_order_is_refused(tmp_path):
    cache, table = build(tmp_path)
    reordered = table.iloc[::-1].reset_index(drop=True)

    with pytest.raises(ValueError, match="different table"):
        cache.reader("train", reordered)
    with pytest.raises(ValueError, match="different table"):
        cache.build("train", reordered, read_sample=fake_read_sample, progress=quiet)


def test_the_fingerprint_changes_when_the_two_halves_swap_places():
    l1c = pd.DataFrame({"tortilla:id": ["a", "b"], "internal:path": ["l1c/a", "l1c/b"]})
    l2a = pd.DataFrame({"tortilla:id": ["a", "b"], "internal:path": ["l2a/a", "l2a/b"]})

    one = table_fingerprint(pd.concat([l1c, l2a], ignore_index=True))
    swapped = table_fingerprint(pd.concat([l2a, l1c], ignore_index=True))

    assert one != swapped


def test_a_missing_cache_asks_to_be_built_first(tmp_path):
    with pytest.raises(FileNotFoundError, match="build it first"):
        LocalCache(tmp_path / "none").reader("train", make_table())


def test_the_cache_refuses_to_start_without_enough_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cache_module.shutil, "disk_usage",
        lambda path: type("U", (), {"free": 1000})(),
    )
    with pytest.raises(RuntimeError, match="GB free"):
        LocalCache(tmp_path / "cache").build(
            "train", make_table(), read_sample=fake_read_sample, progress=quiet
        )


def test_cropping_off_is_refused_because_the_cache_holds_cropped_samples(tmp_path):
    cache, table = build(tmp_path)

    with pytest.raises(ValueError, match="valid window"):
        cache.reader("train", table)(table, 0, False)


def test_a_subset_reader_maps_slice_positions_to_cached_positions(tmp_path):
    cache, table = build(tmp_path)
    reader = cache.reader("train", table)
    smoke = subset_reader(reader, [5, 0, 3])

    assert smoke(None, 0).identifier == "scene-005"
    assert smoke(None, 1).identifier == "scene-000"
    assert smoke(None, 2).identifier == "scene-003"


def test_samples_iterates_the_whole_table_in_order(tmp_path):
    cache, table = build(tmp_path)

    identifiers = [s.identifier for s in cache.samples("train", table)]

    assert identifiers == [f"scene-{i:03d}" for i in range(len(table))]