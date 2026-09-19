import numpy as np

from cloudband.train.protocol import ocm_native_protocol, ocm_shared_protocol
from cloudband.train.resampling import (
    apply_mixed_resolution,
    resample_annotation,
    resample_image,
    sample_gsd,
)


def test_resample_image_matches_the_target_size():
    image = np.random.rand(3, 509, 509).astype(np.float32)
    resized = resample_image(image, target_size=231, nodata_value=None)
    assert resized.shape == (3, 231, 231)


def test_resample_image_accepts_a_batch():
    batch = np.random.rand(4, 3, 509, 509).astype(np.float32)
    resized = resample_image(batch, target_size=339, nodata_value=None)
    assert resized.shape == (4, 3, 339, 339)


def test_resample_annotation_matches_the_target_size():
    annotation = np.zeros((509, 509), dtype=np.int64)
    resized = resample_annotation(annotation, target_size=231)
    assert resized.shape == (231, 231)


def test_resample_annotation_never_invents_class_values():
    annotation = np.random.randint(0, 4, size=(509, 509)).astype(np.int64)
    resized = resample_annotation(annotation, target_size=231)
    assert set(np.unique(resized)) <= set(np.unique(annotation))


def test_resample_image_restamps_nodata_after_bilinear_blending():
    image = np.full((1, 509, 509), 50.0, dtype=np.float32)
    image[:, :50, :50] = 99.0
    resized = resample_image(image, target_size=231, nodata_value=99.0)
    assert 99.0 in np.unique(resized)
    valid = resized[resized != 99.0]
    assert np.allclose(valid, 50.0, atol=1e-3)


def test_sample_gsd_stays_within_bounds():
    rng = np.random.default_rng(0)
    for _ in range(1000):
        gsd = sample_gsd(min_gsd_m=9.0, max_gsd_m=22.0, rng=rng)
        assert 9.0 <= gsd <= 22.0


def test_apply_mixed_resolution_keeps_images_and_annotations_the_same_size():
    rng = np.random.default_rng(0)
    protocol = ocm_shared_protocol(learning_rate=1e-4)
    images = np.random.rand(2, 3, 509, 509).astype(np.float32)
    annotations = np.random.randint(0, 4, size=(2, 509, 509)).astype(np.int64)

    batch = apply_mixed_resolution(
        images, annotations, protocol.min_gsd_m, protocol.max_gsd_m, rng
    )

    assert batch.images.shape[-2:] == (batch.size_px, batch.size_px)
    assert batch.annotations.shape[-2:] == (batch.size_px, batch.size_px)
    assert protocol.min_gsd_m <= batch.gsd_m <= protocol.max_gsd_m


def test_apply_mixed_resolution_can_produce_a_size_below_the_swin_window_limit():
    rng = np.random.default_rng(0)
    protocol = ocm_native_protocol()
    images = np.random.rand(1, 3, 509, 509).astype(np.float32)
    annotations = np.random.randint(0, 4, size=(1, 509, 509)).astype(np.int64)

    sizes = {
        apply_mixed_resolution(
            images, annotations, protocol.min_gsd_m, protocol.max_gsd_m, rng
        ).size_px
        for _ in range(200)
    }

    assert min(sizes) < 224