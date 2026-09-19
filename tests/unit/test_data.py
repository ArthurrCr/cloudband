import numpy as np

from cloudband.datasets.cloudsen12 import Sample
from cloudband.pipelines.phase0 import select_rgn
from cloudband.train.data import CloudSen12Dataset, build_dataloaders
from cloudband.train.normalize import dynamic_z_score

STACK_CHANNELS = 13
PATCH_SIZE = 509


def fake_read_sample(table, index, crop_to_valid=True):
    rng = np.random.default_rng(index)
    image = rng.integers(0, 10000, size=(STACK_CHANNELS, PATCH_SIZE, PATCH_SIZE))
    annotation = rng.integers(0, 4, size=(PATCH_SIZE, PATCH_SIZE))
    return Sample(
        identifier=f"scene-{index}",
        image=image.astype(np.int32),
        annotation=annotation.astype(np.int32),
    )


def test_dataset_selects_only_the_rgn_channels():
    dataset = CloudSen12Dataset(table=range(4), read_sample=fake_read_sample)
    image, annotation = dataset[0]
    assert image.shape == (3, PATCH_SIZE, PATCH_SIZE)
    assert annotation.shape == (PATCH_SIZE, PATCH_SIZE)


def test_dataset_selects_the_same_channels_as_phase0():
    dataset = CloudSen12Dataset(table=range(4), read_sample=fake_read_sample)
    image, _ = dataset[0]
    full_stack = fake_read_sample(range(4), 0).image
    expected = dynamic_z_score(select_rgn(full_stack).astype(np.float32))
    assert np.allclose(image.numpy(), expected)


def test_dataset_length_matches_the_table():
    dataset = CloudSen12Dataset(table=range(7), read_sample=fake_read_sample)
    assert len(dataset) == 7


def test_build_dataloaders_yields_correctly_shaped_batches():
    dls = build_dataloaders(
        train_table=range(6),
        valid_table=range(4),
        micro_batch_size=2,
        read_sample=fake_read_sample,
    )
    images, annotations = next(iter(dls.train))
    assert images.shape == (2, 3, PATCH_SIZE, PATCH_SIZE)
    assert annotations.shape == (2, PATCH_SIZE, PATCH_SIZE)


def test_build_dataloaders_keeps_train_and_valid_separate():
    dls = build_dataloaders(
        train_table=range(6),
        valid_table=range(4),
        micro_batch_size=2,
        read_sample=fake_read_sample,
    )
    assert len(dls.train.dataset) == 6
    assert len(dls.valid.dataset) == 4