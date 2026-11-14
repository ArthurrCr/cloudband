"""Phase 2 test evaluation, as the evaluation notebook runs it.

The notebook calls these functions in order and displays what they return:
cache the test patches, score the published OmniCloudMask (model 4.0) and every
finished seed of the trained models, build the reports of both processing levels,
save them, and draw the figures. Scores are saved to the shared folder as
confusion counts, so a second run reads them instead of scoring again.
"""

from __future__ import annotations

import gc
import re
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure

from cloudband.baselines import ocm
from cloudband.datasets.cloudsen12 import read_sample as gdal_read_sample
from cloudband.eval import phase2 as evaluation
from cloudband.eval import plots
from cloudband.pipelines import phase0
from cloudband.train.cache import LocalCache
from cloudband.train.loop import checkpoint_name
from cloudband.train.protocol import (
    PROTOCOL_REVISION,
    TrainProtocol,
    ocm_shared_protocol,
    swin_shared_protocol,
)

PUBLISHED = "ocm published"
PUBLISHED_VERSION = ocm.LATEST_MODEL_VERSION
TRAINED = ("regnety", "convnext", "ocm", "swin")
ENSEMBLES = {"ocm": ("regnety", "convnext")}

# Split name in the cache and in file names, and the level it holds. L1C is the
# primary result; L2A is the same scenes, reported apart as a robustness check.
LEVELS = {"test": "L1C", "test_l2a": "L2A"}

# Where phase 0 saves the published versions of each level.
DATASET_IDS = {"test": phase0.CLOUDSEN12_TEST, "test_l2a": phase0.CLOUDSEN12_TEST_L2A}

VERSUS_SWIN = (
    "OCM vs Swin, per ROI",
    "regnety alone vs Swin, per ROI",
    "convnext alone vs Swin, per ROI",
)
VERSUS_PUBLISHED = tuple(
    f"{name} vs {PUBLISHED}, per ROI" for name in ("OCM", "regnety alone", "convnext alone", "Swin")
)

Log = Callable[[str], None]


def protocols() -> tuple[TrainProtocol, TrainProtocol]:
    """The OCM and Swin protocols of the comparison.

    Checkpoint names come from the run id and the seed only, so the learning rate
    given here, which differs from the one each network trained with, plays no part.
    """
    return ocm_shared_protocol(learning_rate=1e-4), swin_shared_protocol(learning_rate=1e-4)


def checkpoint_names(seed: int) -> dict[str, str]:
    ocm_protocol, swin_protocol = protocols()
    regnety, convnext = _backbones()
    return {
        "regnety": checkpoint_name(ocm_protocol, seed, regnety),
        "convnext": checkpoint_name(ocm_protocol, seed, convnext),
        "swin": checkpoint_name(swin_protocol, seed),
    }


def _backbones() -> tuple[str, str]:
    from cloudband.models.ocm import PAPER_BACKBONES

    regnety, convnext = PAPER_BACKBONES
    return regnety, convnext


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


def score_published_versions(
    cache: LocalCache,
    tables: Mapping[str, pd.DataFrame],
    versions: tuple[float, ...],
    baseline_dir: Path,
    package_versions: dict[str, str] | None = None,
    progress: Callable[[int, int], None] | None = None,
    log: Log = print,
) -> dict[str, dict[str, dict[float, pd.DataFrame]]]:
    """Every published weight set at each level, read from the cache when not saved.

    Returns, per level name (L1C, L2A), the pooled "scores" and the paper-style
    "paper" table of each version.
    """
    results = {}
    for split, table in tables.items():
        log(f"--- {LEVELS[split]} ---")
        scores, paper = phase0.score_published_versions(
            table,
            versions,
            baseline_dir,
            package_versions=package_versions,
            progress=progress,
            log=log,
            dataset_id=DATASET_IDS[split],
            samples=lambda split=split, table=table: cache.samples(split, table),
        )
        results[LEVELS[split]] = {"scores": scores, "paper": paper}
    return results


def version_table(
    versions: Mapping[str, Mapping[str, Mapping[float, pd.DataFrame]]], kind: str
) -> pd.DataFrame:
    """One table of every level and version: kind is "scores" (pooled) or "paper"."""
    return pd.concat(
        {
            (level, phase0.version_label(version)): frame
            for level, tables in versions.items()
            for version, frame in tables[kind].items()
        },
        names=["level", "version", "experiment"],
    )


def version_comparison(
    versions: Mapping[str, Mapping[str, Mapping[float, pd.DataFrame]]],
    column: str = "median_boa",
) -> dict[str, pd.DataFrame]:
    """Per level, one row per experiment and one column per version, of one metric."""
    return {
        level: pd.DataFrame(
            {phase0.version_label(v): frame[column] for v, frame in tables["paper"].items()}
        )
        for level, tables in versions.items()
    }


def score_published(
    cache: LocalCache,
    tables: Mapping[str, pd.DataFrame],
    results_dir: Path,
    version: float = PUBLISHED_VERSION,
    log: Log = print,
) -> dict[str, pd.DataFrame]:
    """Counts of the published OmniCloudMask on the same cached patches."""
    model_id = phase0.published_model_id(version)
    config = phase0.published_config(version)
    published = {}
    for split, table in tables.items():
        path = counts_path(results_dir, split, model_id)
        if path.exists():
            published[split] = evaluation.load_counts(path)
            continue
        frame = evaluation.score_predictor(
            cache.samples(split, table),
            lambda stack: ocm.predict_array(stack, config),
            PUBLISHED,
        )
        evaluation.save_counts(frame, path)
        published[split] = frame
        log(f"{model_id}, {split}: scored and saved")
    return published


def published_gap(
    published: Mapping[str, pd.DataFrame], baseline_dir: Path, version: float = PUBLISHED_VERSION
) -> float | None:
    """Largest gap between the per-patch BOA from the cache and the one phase 0 saved.

    Phase 0 streams the patches from the remote file; here they come from the local
    cache. The same weights on the same pixels must give the same BOA.
    """
    path = (
        Path(baseline_dir) / phase0.published_model_id(version) / phase0.CLOUDSEN12_TEST
        / "per_scene_boa.csv"
    )
    if not path.exists():
        return None
    return evaluation.legacy_check(published["test"], PUBLISHED, pd.read_csv(path, index_col=0))


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


def with_published(
    counts: Mapping[str, Mapping[int, pd.DataFrame]], published: Mapping[str, pd.DataFrame]
) -> dict[str, dict[int, pd.DataFrame]]:
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
        LEVELS[split]: evaluation.report(by_seed, published=PUBLISHED)
        for split, by_seed in combined.items()
    }


def show_reports(reports: Mapping[str, Mapping[str, pd.DataFrame]], display, decimals=4) -> None:
    if not reports:
        print("no seed is finished for every model yet")
    for level, tables in reports.items():
        role = "primary" if level == "L1C" else "secondary, never pooled with L1C"
        print(f"\n=== {level} test ({role}) ===")
        for title, table in tables.items():
            print(f"\n{title}")
            display(table.round(decimals))


def level_agreement(reports: Mapping[str, Mapping[str, pd.DataFrame]], title: str = VERSUS_SWIN[0]):
    """Whether the mean difference has the same sign at L1C and L2A, per experiment."""
    if not {"L1C", "L2A"} <= set(reports):
        return None
    first, second = (reports[level][title]["mean diff"] for level in ("L1C", "L2A"))
    return (np.sign(first) == np.sign(second)).to_dict()


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def save_reports(reports: Mapping[str, Mapping[str, pd.DataFrame]], results_dir: Path) -> list[Path]:
    split_of = {level: split for split, level in LEVELS.items()}
    paths = []
    for level, tables in reports.items():
        for title, table in tables.items():
            path = (
                Path(results_dir)
                / f"phase2-{PROTOCOL_REVISION}-report-{split_of[level]}-{_slug(title)}.csv"
            )
            table.to_csv(path)
            paths.append(path)
    return paths


def roi_boa(by_seed: Mapping[int, pd.DataFrame], models: Iterable[str]) -> dict[str, pd.DataFrame]:
    """Per-ROI BOA of each model, averaged over seeds."""
    return {model: evaluation.seed_averaged(by_seed, model, "roi") for model in models}


def build_figures(
    reports: Mapping[str, Mapping[str, pd.DataFrame]],
    combined: Mapping[str, Mapping[int, pd.DataFrame]],
) -> dict[str, Figure]:
    """Every figure of the comparison, keyed by the file name it is saved under."""
    if not reports:
        return {}
    figures: dict[str, Figure] = {}
    levels = " and ".join(reports)
    published_label = f"published OmniCloudMask {PUBLISHED_VERSION:.1f}"

    seed_tables = {level: tables["whole split, per seed"] for level, tables in reports.items()}
    figures["mean-vs-published"] = plots.plot_mean_vs_published(
        seed_tables, TRAINED, PUBLISHED,
        f"Trained models, mean of the seeds, against the {published_label}",
    )
    figures["differences-vs-published-per-roi"] = plots.plot_paired_differences(
        reports, VERSUS_PUBLISHED,
        f"Trained models minus the {published_label}, per ROI, {levels}",
    )
    figures["differences-vs-swin-per-roi"] = plots.plot_paired_differences(
        reports, VERSUS_SWIN, f"OCM and its backbones minus Swin, per ROI, {levels}"
    )
    for split, by_seed in combined.items():
        level = LEVELS[split]
        per_roi = roi_boa(by_seed, ("ocm", "swin", PUBLISHED))
        figures[f"roi-vs-published-{split}"] = plots.plot_unit_scatter(
            {"OCM": per_roi["ocm"], "Swin": per_roi["swin"]}, per_roi[PUBLISHED],
            f"published OCM {PUBLISHED_VERSION:.1f}",
            f"BOA per ROI, trained models (seed mean) against the {published_label}, {level} test",
        )
        figures[f"boa-per-seed-{split}"] = plots.plot_seed_boa(
            reports[level]["whole split, per seed"], f"Whole-split BOA per seed, {level} test"
        )
        figures[f"ocm-minus-swin-per-roi-{split}"] = plots.plot_unit_differences(
            evaluation.unit_differences(by_seed, "ocm", "swin", "roi"),
            "OCM", "Swin", f"OCM − Swin per ROI, {level} test",
        )
    return figures


def save_figures(figures: Mapping[str, Figure], directory: Path, dpi: int = 200) -> list[Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, figure in figures.items():
        path = directory / f"phase2-{PROTOCOL_REVISION}-{name}.png"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    return paths
