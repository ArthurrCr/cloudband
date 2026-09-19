"""Shared training protocol for the OCM vs Swin+UPerNet comparison."""

from dataclasses import dataclass, field, fields

from cloudband.datasets.cloudsen12 import VALID_SIZE
from cloudband.pipelines.phase0 import RGN_BANDS, RGN_INDICES

NATIVE_GSD_M = 10.0
PATCH_EXTENT_M = VALID_SIZE * NATIVE_GSD_M
SWIN_WINDOW_SIZE = 7
SWIN_STAGE_DOWNSAMPLE = 32
MIN_SWIN_PATCH_SIZE_PX = SWIN_WINDOW_SIZE * SWIN_STAGE_DOWNSAMPLE

SHARED_MIN_GSD_M = 9.0
SHARED_MAX_GSD_M = 22.0
NATIVE_MIN_GSD_M = 9.0
NATIVE_MAX_GSD_M = 50.0

LR_SEARCH_GRID = (5e-5, 1e-4, 5e-4, 1e-3, 5e-3)
PROXY_FROZEN_EPOCHS = 10
PROXY_UNFROZEN_EPOCHS = 10
FULL_FROZEN_EPOCHS = 50
FULL_UNFROZEN_EPOCHS = 50

COUPLED_FIELDS = frozenset({"run_id", "architecture", "learning_rate"})


def patch_size_px(gsd_m: float) -> int:
    return int(PATCH_EXTENT_M / gsd_m)


def is_swin_viable(gsd_m: float) -> bool:
    return patch_size_px(gsd_m) >= MIN_SWIN_PATCH_SIZE_PX


@dataclass(frozen=True)
class AugmentationConfig:
    image_tear_prob: float = 0.10
    random_rectangle_prob: float = 0.60
    scene_edge_prob: float = 0.10
    rotation_flip_prob: float = 1.00


@dataclass(frozen=True)
class TrainProtocol:
    run_id: str
    architecture: str
    learning_rate: float
    min_gsd_m: float = SHARED_MIN_GSD_M
    max_gsd_m: float = SHARED_MAX_GSD_M
    bands: tuple = RGN_BANDS
    band_indices: tuple = RGN_INDICES
    train_patches: int = 8490
    val_patches: int = 535
    test_patches: int = 975
    processing_levels: tuple = ("L1C", "L2A")
    loss: str = "cross_entropy"
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    effective_batch_size: int = 128
    checkpoint_metric: str = "val_cross_entropy"
    weight_decay: float = 0.01
    normalization: str = "dynamic_z_score"
    pretrained_source: str = "imagenet"
    seeds: tuple = (0, 1, 2, 3, 4)
    scheduler: str = "one_cycle"
    frozen_epochs: int = FULL_FROZEN_EPOCHS
    unfrozen_epochs: int = FULL_UNFROZEN_EPOCHS


def ocm_shared_protocol(learning_rate: float) -> TrainProtocol:
    return TrainProtocol(
        run_id="ocm-rgn-cs12-shared",
        architecture="ocm",
        learning_rate=learning_rate,
    )


def swin_shared_protocol(learning_rate: float) -> TrainProtocol:
    return TrainProtocol(
        run_id="swin-rgn-cs12-shared",
        architecture="swin",
        learning_rate=learning_rate,
    )


def ocm_native_protocol() -> TrainProtocol:
    return TrainProtocol(
        run_id="ocm-rgn-cs12-native",
        architecture="ocm",
        learning_rate=0.005,
        min_gsd_m=NATIVE_MIN_GSD_M,
        max_gsd_m=NATIVE_MAX_GSD_M,
    )


def diverging_fields(left: TrainProtocol, right: TrainProtocol) -> set:
    return {
        f.name
        for f in fields(TrainProtocol)
        if getattr(left, f.name) != getattr(right, f.name)
    }


@dataclass(frozen=True)
class LrSearchRun:
    run_id: str
    learning_rate: float
    val_loss: float
    split: str = "validation"

    def __post_init__(self):
        if self.split != "validation":
            raise ValueError("lr search runs must only use the validation split")


@dataclass(frozen=True)
class LrSearchManifest:
    architecture: str
    runs: tuple = ()
    grid: tuple = LR_SEARCH_GRID

    def winner(self) -> LrSearchRun:
        return min(self.runs, key=lambda run: run.val_loss)