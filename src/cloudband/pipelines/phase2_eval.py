"""Phase 2 test evaluation, as the evaluation notebook runs it.

The notebook calls these functions in order and displays what they return. The test
patches are cached once; every published OmniCloudMask version and every finished
seed of the trained models is scored on them once per processing level, and the
confusion counts are saved to the shared folder. Every table and figure is built
from those counts, so a second run reads them instead of scoring again.

L1C is the primary result. L2A is the same scenes, reported apart as a robustness
check and never pooled with L1C.
"""

from __future__ import annotations

import gc
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure

from cloudband.baselines import ocm
from cloudband.datasets.cloudsen12 import read_sample as gdal_read_sample
from cloudband.eval import phase2 as evaluation
from cloudband.eval import plots
from cloudband.eval.confusion import BinaryConfusion
from cloudband.eval.experiments import pool
from cloudband.eval.report import as_percentages, cloudsen12_table, to_frame
from cloudband.pipelines import phase0
from cloudband.train.cache import LocalCache
from cloudband.train.loop import checkpoint_name
from cloudband.train.protocol import (
    PROTOCOL_REVISION,
    TrainProtocol,
    ocm_shared_protocol,
    swin_shared_protocol,
)

PUBLISHED_VERSIONS = (ocm.PAPER_MODEL_VERSION, 2.0, 3.0, ocm.LATEST_MODEL_VERSION)
TRAINED = ("regnety", "convnext", "ocm", "swin")
ENSEMBLES = {"ocm": ("regnety", "convnext")}

# Split name in the cache and in file names, and the level it holds.
LEVELS = {"test": "L1C", "test_l2a": "L2A"}
SPLITS = {level: split for split, level in LEVELS.items()}

VERSUS_SWIN = (
    "OCM vs Swin, per ROI",
    "regnety alone vs Swin, per ROI",
    "convnext alone vs Swin, per ROI",
)

Log = Callable[[str], None]


def published_name(version: float) -> str:
    return f"ocm published {version:.1f}"


PUBLISHED = tuple(published_name(version) for version in PUBLISHED_VERSIONS)

LABELS = {
    "regnety": "regnety trained",
    "convnext": "convnext trained",
    "ocm": "OCM trained (ensemble)",
    "swin": "Swin-T + UPerNet trained",
    **{published_name(v): f"OCM published {v:.1f}" for v in PUBLISHED_VERSIONS},
}
# The class whose share sorts the patches into ranges, per experiment.
SHARE_CLASSES = {
    "clear": "clear",
    "cloud": "cloud (thick + thin)",
    "shadow": "cloud shadow",
}


# ---- trained runs ---------------------------------------------------------------


def protocols() -> tuple[TrainProtocol, TrainProtocol]:
    """The OCM and Swin protocols of the comparison.

    Checkpoint names come from the run id and the seed only, so the learning rate
    given here, which differs from the one each network trained with, plays no part.
    """
    return ocm_shared_protocol(learning_rate=1e-4), swin_shared_protocol(learning_rate=1e-4)


def _backbones() -> tuple[str, str]:
    from cloudband.models.ocm import PAPER_BACKBONES

    regnety, convnext = PAPER_BACKBONES
    return regnety, convnext


def checkpoint_names(seed: int) -> dict[str, str]:
    ocm_protocol, swin_protocol = protocols()
    regnety, convnext = _backbones()
    return {
        "regnety": checkpoint_name(ocm_protocol, seed, regnety),
        "convnext": checkpoint_name(ocm_protocol, seed, convnext),
        "swin": checkpoint_name(swin_protocol, seed),
    }


def seeds_ready(store) -> list[int]:
    """Seeds whose three networks have all finished training."""
    ocm_protocol, _ = protocols()
    return [
        seed
        for seed in ocm_protocol.seeds
        if all(store.is_done(name) for name in checkpoint_names(seed).values())
    ]


def load_members(store, seed: int, device: str) -> dict[str, torch.nn.Module]:
    """The two OCM backbones and Swin of one seed, from their best checkpoints."""
    from cloudband.models.ocm import build_unet
    from cloudband.models.swin_upernet import build_swin_upernet
    from cloudband.train.store import load_model_from_checkpoint

    regnety, convnext = _backbones()
    builders = {
        "regnety": lambda: build_unet(regnety, pretrained=False),
        "convnext": lambda: build_unet(convnext, pretrained=False),
        "swin": lambda: build_swin_upernet(pretrained=False),
    }
    return {
        model: load_model_from_checkpoint(builders[model], store.checkpoint_path(name), device)
        for model, name in checkpoint_names(seed).items()
    }


# ---- test data ------------------------------------------------------------------


def counts_path(results_dir: Path, split: str, tag: str) -> Path:
    """Saved counts of one split, named by protocol and crop, so a new crop scores again."""
    from cloudband.datasets.cloudsen12 import CROP_REVISION

    return (
        Path(results_dir)
        / f"phase2-{PROTOCOL_REVISION}-crop{CROP_REVISION}-counts-{split}-{tag}.csv"
    )


def load_test_tables() -> dict[str, pd.DataFrame]:
    """The test split at L1C and at L2A, after checking the split sizes."""
    from cloudband.datasets import cloudsen12 as dataset

    sources = {"test": dataset.REMOTE_L1C, "test_l2a": dataset.REMOTE_L2A}
    return {split: phase0.load_test_split(source) for split, source in sources.items()}


def check_fast_reader(
    reader,
    tables: Mapping[str, pd.DataFrame],
    band_indices: Sequence[int] | None = phase0.RGN_INDICES,
) -> int:
    """The fast reader must return what GDAL returns, bands selected.

    Checked on the first and last patch of the first table and the first patch of
    the others, so both ends of the file and every level are covered. band_indices
    are the bands the reader keeps (R-G-NIR by default); None means all of them.
    """
    checks = []
    for position, table in enumerate(tables.values()):
        checks.append((table, 0))
        if position == 0:
            checks.append((table, len(table) - 1))
    for table, row in checks:
        expected, got = gdal_read_sample(table, row), reader(table, row)
        bands = expected.image if band_indices is None else expected.image[list(band_indices)]
        same = np.array_equal(got.image, bands) and np.array_equal(
            got.annotation, expected.annotation
        )
        if not same:
            raise AssertionError(f"the fast reader differs from GDAL at row {row}")
    return len(checks)


def build_test_cache(
    directory: Path, tables: Mapping[str, pd.DataFrame], workers: int = 32, log: Log = print
) -> LocalCache:
    from cloudband.datasets.fast_read import FastReader

    reader = FastReader()
    log(f"fast reader matches GDAL on {check_fast_reader(reader, tables)} samples")
    from cloudband.datasets.cloudsen12 import CROP_REVISION

    cache = LocalCache(Path(directory) / f"crop{CROP_REVISION}")
    for split, table in tables.items():
        cache.build(split, table, read_sample=reader, workers=workers, bands_selected=True)
    return cache


# ---- scoring --------------------------------------------------------------------


def score_published(
    cache: LocalCache,
    tables: Mapping[str, pd.DataFrame],
    results_dir: Path,
    versions: Sequence[float] = PUBLISHED_VERSIONS,
    log: Log = print,
) -> dict[str, pd.DataFrame]:
    """Counts of every published version on the cached patches, one frame per split.

    Versions run one after the other over all the patches, so the package loads each
    version's weights once.
    """
    published = {}
    for split, table in tables.items():
        frames = []
        for version in versions:
            model_id = phase0.published_model_id(version)
            path = counts_path(results_dir, split, model_id)
            if path.exists():
                frame = evaluation.load_counts(path)
            else:
                config = phase0.published_config(version)
                frame = evaluation.score_predictor(
                    cache.samples(split, table),
                    lambda stack, config=config: ocm.predict_array(stack, config),
                    published_name(version),
                )
                evaluation.save_counts(frame, path)
                log(f"{model_id}, {split}: scored and saved")
            frames.append(frame.assign(model=published_name(version)))
        published[split] = pd.concat(frames, ignore_index=True)
    return published


def crop_check(published: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """evaluation.clear_pixel_audit of each level (DIVERGENCES D17, docs/AUDIT.md A1).

    With the right crop no patch is held above a floor of clear pixels and no count
    repeats across many cloudy patches; with padding kept, both happen.
    """
    return pd.DataFrame(
        {LEVELS[split]: evaluation.clear_pixel_audit(frame) for split, frame in published.items()}
    )


def score_trained(
    cache: LocalCache,
    tables: Mapping[str, pd.DataFrame],
    store,
    results_dir: Path,
    device: str,
    load: Callable[[int], Mapping[str, torch.nn.Module]] | None = None,
    log: Log = print,
) -> dict[str, dict[int, pd.DataFrame]]:
    """Counts of every finished seed, per split; a seed's models load only if needed."""
    load = load or (lambda seed: load_members(store, seed, device))
    ready = seeds_ready(store)
    log(f"seeds finished for every model: {ready} of {list(protocols()[0].seeds)}")

    counts: dict[str, dict[int, pd.DataFrame]] = {split: {} for split in tables}
    for seed in ready:
        members = None
        for split, table in tables.items():
            path = counts_path(results_dir, split, f"seed{seed}")
            if path.exists():
                counts[split][seed] = evaluation.load_counts(path)
                continue
            members = members or load(seed)
            frame = evaluation.score_models(cache.samples(split, table), members, ENSEMBLES, device)
            evaluation.save_counts(frame, path)
            counts[split][seed] = frame
            log(f"seed {seed}, {split}: scored and saved")
        del members
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return counts


# ---- published versions alone ---------------------------------------------------


def _per_scene(counts: pd.DataFrame) -> dict[str, dict[str, BinaryConfusion]]:
    per_scene: dict[str, dict[str, BinaryConfusion]] = {}
    for row in counts.itertuples(index=False):
        per_scene.setdefault(row.scene, {})[row.experiment] = BinaryConfusion(
            row.experiment, tp=int(row.tp), tn=int(row.tn), fp=int(row.fp), fn=int(row.fn)
        )
    return per_scene


def _published_by_level(published: Mapping[str, pd.DataFrame], build) -> pd.DataFrame:
    frames = {}
    for split, frame in published.items():
        for name in PUBLISHED:
            part = frame[frame["model"] == name]
            if not part.empty:
                frames[(LEVELS[split], LABELS[name])] = build(_per_scene(part))
    return pd.concat(frames, names=["level", "model", "experiment"])


def published_scores(published: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Whole-split metrics (percent) of every published version at each level."""
    return _published_by_level(published, lambda per_scene: as_percentages(to_frame(pool(per_scene))))


def published_cloudsen12_tables(published: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """The CloudSEN12 paper's reporting table of every published version at each level."""
    return _published_by_level(published, cloudsen12_table)


# ---- reports --------------------------------------------------------------------


def with_published(
    counts: Mapping[str, Mapping[int, pd.DataFrame]], published: Mapping[str, pd.DataFrame]
) -> dict[str, dict[int, pd.DataFrame]]:
    """Each seed's counts with the published versions added, per split with seeds."""
    return {
        split: evaluation.with_reference(by_seed, published[split])
        for split, by_seed in counts.items()
        if by_seed
    }


def build_reports(
    combined: Mapping[str, Mapping[int, pd.DataFrame]],
) -> dict[str, dict[str, pd.DataFrame]]:
    """evaluation.report of each level with seeds, keyed by level name (L1C, L2A)."""
    return {
        LEVELS[split]: evaluation.report(
            by_seed, published=[n for n in PUBLISHED if n in set(next(iter(by_seed.values()))["model"])]
        )
        for split, by_seed in combined.items()
    }


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def save_reports(reports: Mapping[str, Mapping[str, pd.DataFrame]], results_dir: Path) -> list[Path]:
    paths = []
    for level, tables in reports.items():
        for title, table in tables.items():
            path = (
                Path(results_dir)
                / f"phase2-{PROTOCOL_REVISION}-report-{SPLITS[level]}-{_slug(title)}.csv"
            )
            table.to_csv(path)
            paths.append(path)
    return paths


def _in_points(table: pd.DataFrame) -> pd.DataFrame:
    """BOA columns and differences in percent, counts and p-values as they are."""
    table = table.copy()
    keep = {"n", "p", "p holm", "first better in"} | {
        c for c in table.columns if c.endswith(" better in")
    }
    for column in table.columns:
        if column not in keep:
            table[column] = table[column] * 100
    return table


def table_all_models(reports, level: str) -> pd.DataFrame:
    """Median BOA per patch (percent) of every model: the seeds, their mean and SD."""
    table = _in_points(reports[level]["median per patch, per seed"])
    order = [m for m in (*PUBLISHED, *TRAINED) if m in table.index.get_level_values(0)]
    table = table.reindex(order, level=0)
    return table.rename(index=LABELS, level=0)


def table_ocm_vs_swin(reports, level: str) -> pd.DataFrame:
    """OCM and each backbone minus Swin, per ROI, in BOA points."""
    return _in_points(evaluation.stacked(reports[level], VERSUS_SWIN))


def versus_published_titles(reports, level: str) -> list[str]:
    return [
        f"{model} vs {name}, per ROI"
        for name in PUBLISHED
        for model in ("OCM", "Swin")
        if f"{model} vs {name}, per ROI" in reports[level]
    ]


def table_vs_published(reports, level: str) -> pd.DataFrame:
    """Trained OCM and Swin minus each published version, per ROI, in BOA points."""
    table = _in_points(evaluation.stacked(reports[level], versus_published_titles(reports, level)))
    return table.rename(
        index=lambda name: name.replace(" vs ", " − ").replace(
            "ocm published ", "published "
        ),
        level=0,
    )


def level_agreement(reports, title: str = VERSUS_SWIN[0]):
    """Whether the mean difference has the same sign at L1C and L2A, per experiment."""
    if not {"L1C", "L2A"} <= set(reports):
        return None
    first, second = (reports[level][title]["mean diff"] for level in ("L1C", "L2A"))
    return (np.sign(first) == np.sign(second)).to_dict()


# ---- figures ---------------------------------------------------------------------


def figure_all_models(reports, level: str) -> Figure:
    """Where every model stands: the four published versions and the trained models."""
    table = reports[level]["median per patch, per seed"]
    present = set(table.index.get_level_values(0))
    published = [m for m in PUBLISHED if m in present]
    trained = [m for m in TRAINED if m in present]
    return plots.plot_all_models(
        table, [published, trained], LABELS,
        f"Median BOA per patch of every model, {level} test",
    )


def _share_models(by_seed) -> list[str]:
    present = set(next(iter(by_seed.values()))["model"])
    return [m for m in (*PUBLISHED, *TRAINED) if m in present]


def table_by_share(combined, level: str) -> pd.DataFrame:
    """Median BOA per patch (percent) of every model in each range of class share.

    The value is the mean over seeds of each seed's median; patches counts the
    patches with a defined BOA in the range.
    """
    by_seed = combined[SPLITS[level]]
    table = evaluation.boa_by_share(by_seed)
    wide = (table["mean"] * 100).unstack("model")[_share_models(by_seed)]
    wide = wide.rename(columns=LABELS)
    wide.insert(0, "patches", table["n"].groupby(level=["experiment", "bin"]).first())
    wide.index.names = ["experiment", "class share"]
    return wide


def figure_by_share(combined, level: str, experiment: str) -> Figure:
    """BOA of every model per range of the experiment's class share, one experiment."""
    by_seed = combined[SPLITS[level]]
    table = evaluation.boa_by_share(by_seed).loc[experiment]
    return plots.plot_boa_by_share(
        table, _share_models(by_seed), LABELS,
        f"{experiment}: BOA by share of {SHARE_CLASSES[experiment]} pixels in the patch, "
        f"{level} test",
    )


def figure_path(directory: Path, name: str) -> Path:
    return Path(directory) / f"phase2-{PROTOCOL_REVISION}-{name}.png"


def show(figure: Figure, directory: Path, name: str, display, dpi: int = 200) -> None:
    """Save the figure as PNG and display it."""
    path = figure_path(directory, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    display(figure)
