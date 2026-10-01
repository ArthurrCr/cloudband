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

# Learning-rate search grid: five points about half a decade apart in log10 (the
# usual 1 and 3 per decade, so the steps are 0.48 and 0.52 rather than exactly 0.5).
#
# Protocol. Each architecture is searched on the same space with the same budget,
# the winner is chosen on the validation split only, and it is then retrained with
# several seeds. This is the GEO-Bench-2 protocol (Simumba et al., 2025, arXiv:
# 2511.15658, sections 3.1 and 6.2), which includes CloudSEN12+ with the same
# 535-patch validation and 975-patch test splits. It allows up to 16 trials per
# model; five is what the cost of a proxy run affords here.
#
# Range. GEO-Bench-2 searches 1e-6 to 1e-3 for fine-tuning pretrained backbones on
# Earth-observation segmentation (AdamW, weight decay 0.01). Swin + UPerNet is
# published at 6e-5 (Liu et al., 2021, ICCV, ADE20K), and OCM's own training uses
# base_lr = 1e-3 in the same fastai fine_tune call. The grid keeps the upper part of
# the GEO-Bench-2 range and goes one step past it. fastai's fine_tune gives the
# pretrained layers base_lr / 200 up to base_lr / 2 after unfreezing, so base_lr is
# closer to the head's rate than to a single rate for the whole network. That
# reasoning is this project's, not the literature's.
#
# Whether the optimum falls inside is checked, not assumed:
# LrSearchManifest.winner_at_edge flags a winner on either end of the grid, and the
# search then has to be extended for both architectures and run again.
LR_SEARCH_GRID = (3e-5, 1e-4, 3e-4, 1e-3, 3e-3)
# Epoch budget, the same for both architectures (ADR-0023 D5). The full run
# follows the OmniCloudMask training notebook (training/Train OCM models.ipynb,
# the non-demo branch: freeze_epochs = 15, unfrozen_epochs = 15). A proxy run is a
# third of that, so the search costs less than the runs it serves; it has not
# been checked against how the final ranking of learning rates would come out.
# The validation loss is recorded every epoch so convergence can be checked
# afterwards instead of assumed.
PROXY_FROZEN_EPOCHS = 5
PROXY_UNFROZEN_EPOCHS = 5
FULL_FROZEN_EPOCHS = 15
FULL_UNFROZEN_EPOCHS = 15

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
    # fp16 autocast with loss scaling, used only when a GPU is present. Measured
    # about 2.3x faster per step on a T4. It is a systems choice, identical for
    # both architectures, and it changes the numerics slightly.
    mixed_precision: bool = True


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

    def winner_at_edge(self) -> bool:
        """True when the best candidate is the lowest or the highest of the grid.

        The best rate may then lie outside the range searched. A grid with fewer
        than three points has no interior, so nothing can be said and this is
        False.
        """
        if len(self.grid) < 3:
            return False
        return self.winner().learning_rate in (min(self.grid), max(self.grid))