import inspect

import pytest

from cloudband.train.protocol import (
    COUPLED_FIELDS,
    LR_SEARCH_GRID,
    LrSearchManifest,
    LrSearchRun,
    diverging_fields,
    is_swin_viable,
    ocm_native_protocol,
    ocm_shared_protocol,
    patch_size_px,
    swin_shared_protocol,
)


def test_t_arch_02_shared_protocol_builders_only_expose_learning_rate():
    ocm_params = set(inspect.signature(ocm_shared_protocol).parameters)
    swin_params = set(inspect.signature(swin_shared_protocol).parameters)
    assert ocm_params == {"learning_rate"}
    assert swin_params == {"learning_rate"}


def test_t_arch_04_gsd_range_matches_between_shared_configs():
    ocm = ocm_shared_protocol(learning_rate=1e-4)
    swin = swin_shared_protocol(learning_rate=5e-4)
    assert ocm.min_gsd_m == swin.min_gsd_m
    assert ocm.max_gsd_m == swin.max_gsd_m


def test_t_arch_05_lr_grid_and_run_count_match_between_architectures():
    ocm_runs = [ocm_shared_protocol(lr) for lr in LR_SEARCH_GRID]
    swin_runs = [swin_shared_protocol(lr) for lr in LR_SEARCH_GRID]
    assert len(ocm_runs) == len(swin_runs)
    ocm_rates = [run.learning_rate for run in ocm_runs]
    swin_rates = [run.learning_rate for run in swin_runs]
    assert ocm_rates == swin_rates == list(LR_SEARCH_GRID)


def test_t_arch_06_pretrained_source_matches_between_architectures():
    ocm = ocm_shared_protocol(learning_rate=1e-4)
    swin = swin_shared_protocol(learning_rate=1e-4)
    assert ocm.pretrained_source == swin.pretrained_source == "imagenet"


def test_t_arch_07_only_coupled_fields_differ_between_shared_configs():
    ocm = ocm_shared_protocol(learning_rate=1e-4)
    swin = swin_shared_protocol(learning_rate=5e-4)
    assert diverging_fields(ocm, swin) <= COUPLED_FIELDS


def test_t_arch_08_lr_search_manifest_only_accepts_validation_runs():
    manifest = LrSearchManifest(
        architecture="ocm",
        runs=(
            LrSearchRun(
                run_id="ocm-rgn-cs12-shared",
                learning_rate=1e-4,
                val_loss=0.31,
            ),
        ),
    )
    assert all(run.split == "validation" for run in manifest.runs)


def test_t_arch_08_lr_search_run_rejects_test_split():
    with pytest.raises(ValueError):
        LrSearchRun(
            run_id="ocm-rgn-cs12-shared",
            learning_rate=1e-4,
            val_loss=0.31,
            split="test",
        )


def test_t_arch_09_scheduler_shape_matches_between_architectures():
    ocm = ocm_shared_protocol(learning_rate=1e-4)
    swin = swin_shared_protocol(learning_rate=5e-4)
    assert ocm.scheduler == swin.scheduler
    assert ocm.frozen_epochs == swin.frozen_epochs
    assert ocm.unfrozen_epochs == swin.unfrozen_epochs


def test_t_geom_04_shared_range_stays_within_swin_window_limit():
    protocol = ocm_shared_protocol(learning_rate=1e-4)
    for gsd in (protocol.min_gsd_m, protocol.max_gsd_m):
        assert patch_size_px(gsd) >= 224
        assert patch_size_px(gsd) // 32 >= 7
        assert is_swin_viable(gsd)


def test_t_geom_04_native_range_exceeds_swin_window_limit_at_coarse_end():
    protocol = ocm_native_protocol()
    assert not is_swin_viable(protocol.max_gsd_m)