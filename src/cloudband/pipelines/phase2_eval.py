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
COMPARISON_LABELS = {
    "OCM vs Swin, per ROI": "OCM ensemble − Swin",
    "regnety alone vs Swin, per ROI": "regnety alone − Swin",
    "convnext alone vs Swin, per ROI": "convnext alone − Swin",
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
    return Path(results_dir) / f"phase2-{PROTOCOL_REVISION}-counts-{split}-{tag}.csv"


def load_test_tables() -> dict[str, pd.DataFrame]:
    """The test split at L1C and at L2A, after checking the split sizes."""
    from cloudband.datasets import cloudsen12 as dataset

    sources = {"test": dataset.REMOTE_L1C, "test_l2a": dataset.REMOTE_L2A}
    return {split: phase0.load_test_split(source) for split, source in sources.items()}


def check_fast_reader(reader, tables: Mapping[str, pd.DataFrame]) -> int:
    """The fast reader must return what GDAL returns, bands selected.

    Checked on the first and last patch of the first table and the first patch of
    the others, so both ends of the file and every level are covered.
    """
    checks = []
    for position, table in enumerate(tables.values()):
        checks.append((table, 0))
        if position == 0:
            checks.append((table, len(table) - 1))
    for table, row in checks:
        expected, got = gdal_read_sample(table, row), reader(table, row)
        same = np.array_equal(got.image, phase0.select_rgn(expected.image)) and np.array_equal(
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
    cache = LocalCache(Path(directory))
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


def published_gaps(published: Mapping[str, pd.DataFrame], baseline_dir: Path) -> pd.DataFrame:
    """Gap to the per-patch BOA phase 0 saved for each version, read from the remote file.

    The same weights on the same pixels must give the same BOA, whether the patch
    came from the remote file or from the local cache.
    """
    rows = []
    for name in sorted(set(published["test"]["model"])):
        version = float(name.rsplit(" ", 1)[1])
        path = (
            Path(baseline_dir) / phase0.published_model_id(version) / phase0.CLOUDSEN12_TEST
            / "per_scene_boa.csv"
        )
        if path.exists():
            gap = evaluation.legacy_check(published["test"], name, pd.read_csv(path, index_col=0))
            rows.append({"model": LABELS[name], "largest gap": gap})
    return pd.DataFrame(rows, columns=["model", "largest gap"])


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


def legacy_gaps(counts: Mapping[int, pd.DataFrame], results_dir: Path) -> pd.DataFrame:
    """Gap to the per-patch BOA saved by the scoring used before, where it exists."""
    rows = []
    for seed, frame in counts.items():
        for model in ("ocm", "swin"):
            path = Path(results_dir) / f"{model}_shared_per_scene_boa_seed{seed}.csv"
            if path.exists():
                gap = evaluation.legacy_check(frame, model, pd.read_csv(path, index_col=0))
                rows.append({"seed": seed, "model": model, "largest gap": gap})
    return pd.DataFrame(rows, columns=["seed", "model", "largest gap"])


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
    """Whole-split BOA (percent) of every model: the seeds, their mean and SD."""
    table = _in_points(reports[level]["whole split, per seed"])
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


# ---- figures, one question each --------------------------------------------------


def figure_all_models(reports, level: str) -> Figure:
    """Where every model stands: the four published versions and the trained models."""
    table = reports[level]["whole split, per seed"]
    present = set(table.index.get_level_values(0))
    published = [m for m in PUBLISHED if m in present]
    trained = [m for m in TRAINED if m in present]
    return plots.plot_all_models(
        table, [published, trained], LABELS,
        f"Whole-split BOA of every model, {level} test",
    )


def figure_ocm_vs_swin(reports, level: str) -> Figure:
    """Trained OCM (and each backbone) against trained Swin, per ROI."""
    return plots.plot_paired_differences(
        {level: reports[level]}, VERSUS_SWIN,
        f"Trained OCM minus trained Swin, per ROI, {level} test",
        labels=COMPARISON_LABELS,
    )


def figure_vs_published(reports, level: str) -> Figure:
    """Trained OCM and Swin against each published version, per ROI."""
    present = [n for n in PUBLISHED if f"OCM vs {n}, per ROI" in reports[level]]
    return plots.plot_versus_references(
        reports[level], {"OCM": "OCM trained", "Swin": "Swin trained"},
        present, LABELS,
        f"Trained models minus each published OmniCloudMask version, per ROI, {level} test",
    )


def figure_roi_wins(combined, level: str) -> Figure:
    """How many ROIs the trained OCM wins over the trained Swin."""
    return plots.plot_unit_differences(
        evaluation.unit_differences(combined[SPLITS[level]], "ocm", "swin", "roi"),
        "OCM", "Swin", f"Trained OCM minus trained Swin, BOA per ROI, {level} test",
    )


def figure_roi_vs_published(combined, level: str, version: float = ocm.LATEST_MODEL_VERSION) -> Figure:
    """Each ROI: trained OCM and Swin (seed mean) against one published version."""
    by_seed = combined[SPLITS[level]]
    name = published_name(version)
    per_roi = {m: evaluation.seed_averaged(by_seed, m, "roi") for m in ("ocm", "swin", name)}
    return plots.plot_unit_scatter(
        {"OCM": per_roi["ocm"], "Swin": per_roi["swin"]}, per_roi[name], LABELS[name],
        f"BOA per ROI: trained models (seed mean) against {LABELS[name]}, {level} test",
    )


def figure_levels(reports) -> Figure:
    """Whether OCM minus Swin holds at both levels."""
    return plots.plot_paired_differences(
        reports, VERSUS_SWIN[:1], "Trained OCM minus trained Swin, per ROI: L1C and L2A",
        labels=COMPARISON_LABELS,
    )


def figure_path(directory: Path, name: str) -> Path:
    return Path(directory) / f"phase2-{PROTOCOL_REVISION}-{name}.png"


def show(figure: Figure, directory: Path, name: str, display, dpi: int = 200) -> None:
    """Save the figure as PNG and display it."""
    path = figure_path(directory, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    display(figure)
