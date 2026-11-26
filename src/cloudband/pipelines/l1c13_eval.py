"""Test evaluation of the 13-band experiment, as its evaluation notebook runs it.

The 13-band models (experiment l1c13, DIVERGENCES D18) are scored on the L1C test
split, the only level that has all 13 bands. Their confusion counts are saved next to
those of the R-G-NIR models and of the published OmniCloudMask versions, which were
scored by the phase 2 evaluation and are only read here. Every table and figure comes
from the counts, so a second run reads them instead of scoring again.

Two comparisons, read differently:

- OCM against Swin, both at 13 bands: the controlled comparison of the experiment,
  same patches, same protocol, same first-layer adaptation.
- 13 bands against R-G-NIR, per architecture: informative only (LIMITATIONS L2). The
  R-G-NIR models also trained on L2A, with the old crop (L1), and without the
  adapted first layer, so the difference is not the effect of the bands alone.

In the counts, the 13-band models keep the phase 2 names (regnety, convnext, ocm,
swin), so the phase 2 report applies unchanged; the R-G-NIR models are renamed with
the suffix " rgn".
"""

from __future__ import annotations

import gc
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pandas as pd
import torch
from matplotlib.figure import Figure

from cloudband.eval import phase2 as evaluation
from cloudband.eval import plots
from cloudband.pipelines import phase0
from cloudband.pipelines.phase2_eval import (
    ENSEMBLES,
    PUBLISHED,
    PUBLISHED_VERSIONS,
    SHARE_CLASSES,
    VERSUS_SWIN,
    _in_points,
    check_fast_reader,
    counts_path,
    published_name,
)
from cloudband.train.cache import LocalCache, bytes_per_sample
from cloudband.train.experiments import experiment
from cloudband.train.loop import checkpoint_name
from cloudband.train.protocol import PROTOCOL_REVISION

EXPERIMENT = experiment("l1c13")
SPLIT = "test"  # the L1C test split
TRAINED = ("regnety", "convnext", "ocm", "swin")
RGN_SUFFIX = " rgn"
RGN = tuple(f"{model}{RGN_SUFFIX}" for model in TRAINED)

LABELS = {
    "regnety": "regnety 13 bands",
    "convnext": "convnext 13 bands",
    "ocm": "OCM 13 bands (ensemble)",
    "swin": "Swin-T + UPerNet 13 bands",
    "regnety rgn": "regnety R-G-NIR",
    "convnext rgn": "convnext R-G-NIR",
    "ocm rgn": "OCM R-G-NIR (ensemble)",
    "swin rgn": "Swin-T + UPerNet R-G-NIR",
    **{published_name(v): f"OCM published {v:.1f}" for v in PUBLISHED_VERSIONS},
}
# Same families as phase 2 (OCM blue, Swin orange); R-G-NIR is the lighter step.
COLORS = {
    "regnety": "#5598e7",
    "convnext": "#5598e7",
    "ocm": "#1c5cab",
    "swin": plots.SWIN_COLOR,
    "regnety rgn": "#a9cbf4",
    "convnext rgn": "#a9cbf4",
    "ocm rgn": "#86b6ef",
    "swin rgn": "#f5a582",
    # the gray of 4.0 in the phase 2 figures, which show all four versions
    published_name(PUBLISHED_VERSIONS[-1]): plots.PUBLISHED_RAMP[-1],
}
# Bars of the share figures: the latest published version, then each architecture at
# R-G-NIR and at 13 bands, side by side.
SHARE_MODELS = (published_name(PUBLISHED_VERSIONS[-1]), "ocm rgn", "ocm", "swin rgn", "swin")

BANDS_VS_RGN = (
    "OCM 13 bands vs OCM R-G-NIR, per ROI",
    "Swin 13 bands vs Swin R-G-NIR, per ROI",
)

Log = Callable[[str], None]


# ---- trained runs ---------------------------------------------------------------


def _backbones() -> tuple[str, str]:
    from cloudband.models.ocm import PAPER_BACKBONES

    regnety, convnext = PAPER_BACKBONES
    return regnety, convnext


def checkpoint_names(seed: int) -> dict[str, str]:
    """The checkpoint of each 13-band network of one seed."""
    ocm_protocol, swin_protocol = EXPERIMENT.protocols()
    regnety, convnext = _backbones()
    return {
        "regnety": checkpoint_name(ocm_protocol, seed, regnety),
        "convnext": checkpoint_name(ocm_protocol, seed, convnext),
        "swin": checkpoint_name(swin_protocol, seed),
    }


def seeds_ready(store) -> list[int]:
    """Seeds whose three 13-band networks have all finished training."""
    ocm_protocol, _ = EXPERIMENT.protocols()
    return [
        seed
        for seed in ocm_protocol.seeds
        if all(store.is_done(name) for name in checkpoint_names(seed).values())
    ]


def load_members(store, seed: int, device: str) -> dict[str, torch.nn.Module]:
    """The two OCM backbones and Swin of one seed, built for 13 input bands."""
    from cloudband.models.ocm import build_unet
    from cloudband.models.swin_upernet import build_swin_upernet
    from cloudband.train.store import load_model_from_checkpoint

    regnety, convnext = _backbones()
    bands = EXPERIMENT.band_count
    builders = {
        "regnety": lambda: build_unet(regnety, pretrained=False, in_chans=bands),
        "convnext": lambda: build_unet(convnext, pretrained=False, in_chans=bands),
        "swin": lambda: build_swin_upernet(pretrained=False, in_chans=bands),
    }
    return {
        model: load_model_from_checkpoint(builders[model], store.checkpoint_path(name), device)
        for model, name in checkpoint_names(seed).items()
    }


def seed_counts_path(results_dir: Path, seed: int) -> Path:
    return counts_path(results_dir, SPLIT, f"{EXPERIMENT.name}-seed{seed}")


# ---- test data ------------------------------------------------------------------


def load_test_table() -> pd.DataFrame:
    """The L1C test split, after checking the split sizes."""
    from cloudband.datasets import cloudsen12 as dataset

    return phase0.load_test_split(dataset.REMOTE_L1C)


def build_test_cache(
    directory: Path, table: pd.DataFrame, workers: int = 32, log: Log = print
) -> LocalCache:
    """Every band of every L1C test patch, after checking the reader against GDAL."""
    from cloudband.datasets.cloudsen12 import CROP_REVISION
    from cloudband.datasets.fast_read import FastReader

    reader = FastReader(bands=EXPERIMENT.reader_bands)
    checked = check_fast_reader(reader, {SPLIT: table}, EXPERIMENT.reader_bands)
    log(f"fast reader matches GDAL on {checked} samples, {EXPERIMENT.band_count} bands")
    cache = LocalCache(Path(directory) / f"{EXPERIMENT.name}-crop{CROP_REVISION}")
    cache.build(
        SPLIT, table, read_sample=reader, workers=workers, bands_selected=True,
        sample_bytes=bytes_per_sample(EXPERIMENT.band_count),
    )
    return cache


# ---- counts ---------------------------------------------------------------------


def score_trained(
    cache: LocalCache,
    table: pd.DataFrame,
    store,
    results_dir: Path,
    device: str,
    load: Callable[[int], Mapping[str, torch.nn.Module]] | None = None,
    log: Log = print,
) -> dict[int, pd.DataFrame]:
    """Counts of every finished 13-band seed on the L1C test; models load only if needed."""
    load = load or (lambda seed: load_members(store, seed, device))
    ready = seeds_ready(store)
    log(f"13-band seeds finished for every network: {ready}")
    counts = {}
    for seed in ready:
        path = seed_counts_path(results_dir, seed)
        if path.exists():
            counts[seed] = evaluation.load_counts(path)
            continue
        members = load(seed)
        frame = evaluation.score_models(cache.samples(SPLIT, table), members, ENSEMBLES, device)
        evaluation.save_counts(frame, path)
        counts[seed] = frame
        log(f"seed {seed}: scored and saved")
        del members
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return counts


def load_rgn_counts(results_dir: Path, seeds: Sequence[int]) -> dict[int, pd.DataFrame]:
    """The phase 2 counts of the R-G-NIR models on the L1C test, renamed with " rgn"."""
    counts = {}
    for seed in seeds:
        path = counts_path(results_dir, SPLIT, f"seed{seed}")
        if not path.exists():
            raise FileNotFoundError(
                f"{path.name} is not in {results_dir}: it comes from the phase 2 evaluation "
                "(notebooks/00_baselines/score_ocm_cloudsen12.ipynb), run it first"
            )
        frame = evaluation.load_counts(path)
        counts[seed] = frame.assign(model=frame["model"] + RGN_SUFFIX)
    return counts


def published_counts_path(results_dir: Path, version: float) -> Path:
    """The saved counts of one published version, whatever package version scored them."""
    stem = counts_path(results_dir, SPLIT, "ocm-rgn-published-v").name.removesuffix(".csv")
    pattern = re.compile(re.escape(stem) + r"[0-9.]+-model" + re.escape(str(version)) + r"\.csv")
    found = sorted(p for p in Path(results_dir).iterdir() if pattern.fullmatch(p.name))
    if len(found) != 1:
        raise FileNotFoundError(
            f"expected one saved count file of published version {version} in {results_dir}, "
            f"found {[p.name for p in found]}; they come from the phase 2 evaluation"
        )
    return found[0]


def load_published(results_dir: Path) -> pd.DataFrame:
    """The phase 2 counts of every published version on the L1C test."""
    frames = [
        evaluation.load_counts(published_counts_path(results_dir, version)).assign(
            model=published_name(version)
        )
        for version in PUBLISHED_VERSIONS
    ]
    return pd.concat(frames, ignore_index=True)


def combine(
    counts: Mapping[int, pd.DataFrame],
    rgn: Mapping[int, pd.DataFrame],
    published: pd.DataFrame,
) -> dict[int, pd.DataFrame]:
    """Per seed: the 13-band models, the R-G-NIR models of the same seed number, and the
    published versions, all on exactly the same patches.

    Seeds of the two experiments are independent draws; the comparisons average each
    model over its seeds before pairing patches or ROIs, so which seed meets which
    plays no part.
    """
    missing = sorted(set(counts) ^ set(rgn))
    if missing:
        raise ValueError(f"seeds {missing} are in one experiment and not in the other")
    joined = {}
    for seed, frame in counts.items():
        if set(frame["scene"]) != set(rgn[seed]["scene"]):
            raise ValueError(f"seed {seed}: 13-band and R-G-NIR counts cover different patches")
        joined[seed] = pd.concat([frame, rgn[seed]], ignore_index=True)
    return evaluation.with_reference(joined, published)


# ---- report ---------------------------------------------------------------------


def build_report(combined: Mapping[int, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """The phase 2 report of the 13-band models, plus 13 bands against R-G-NIR."""
    tables = evaluation.report(combined, published=PUBLISHED)
    for title, (model_a, model_b) in zip(BANDS_VS_RGN, (("ocm", "ocm rgn"), ("swin", "swin rgn"))):
        tables[title] = evaluation.paired_comparison(combined, model_a, model_b, "roi")
    return tables


def save_report(report: Mapping[str, pd.DataFrame], results_dir: Path) -> list[Path]:
    paths = []
    for title, table in report.items():
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        path = (
            Path(results_dir)
            / f"phase2-{PROTOCOL_REVISION}-report-{SPLIT}-{EXPERIMENT.name}-{slug}.csv"
        )
        table.to_csv(path)
        paths.append(path)
    return paths


def table_all_models(report) -> pd.DataFrame:
    """Median BOA per patch (percent) of every model: the seeds, their mean and SD."""
    table = _in_points(report["median per patch, per seed"])
    order = [m for m in (*PUBLISHED, *RGN, *TRAINED) if m in table.index.get_level_values(0)]
    return table.reindex(order, level=0).rename(index=LABELS, level=0)


def table_ocm_vs_swin(report) -> pd.DataFrame:
    """OCM and each backbone minus Swin, per ROI, all at 13 bands, in BOA points."""
    return _in_points(evaluation.stacked(report, VERSUS_SWIN))


def table_bands(report) -> pd.DataFrame:
    """13 bands minus R-G-NIR, per architecture, per ROI, in BOA points (informative)."""
    return _in_points(evaluation.stacked(report, BANDS_VS_RGN))


def table_vs_published(report) -> pd.DataFrame:
    """OCM and Swin at 13 bands minus each published version, per ROI, in BOA points."""
    titles = [
        f"{model} vs {name}, per ROI"
        for name in PUBLISHED
        for model in ("OCM", "Swin")
        if f"{model} vs {name}, per ROI" in report
    ]
    table = _in_points(evaluation.stacked(report, titles))
    return table.rename(
        index=lambda name: name.replace(" vs ", " 13 bands − ").replace(
            "ocm published ", "published "
        ),
        level=0,
    )


# ---- figures --------------------------------------------------------------------


def figure_all_models(report) -> Figure:
    """Where every model stands: published versions, R-G-NIR models, 13-band models."""
    table = report["median per patch, per seed"]
    present = set(table.index.get_level_values(0))
    groups = [[m for m in group if m in present] for group in (PUBLISHED, RGN, TRAINED)]
    return plots.plot_all_models(
        table, [g for g in groups if g], LABELS,
        "Median BOA per patch of every model, L1C test, R-G-NIR and 13 bands",
        colors=COLORS,
    )


def table_by_share(combined, models: Sequence[str] = SHARE_MODELS) -> pd.DataFrame:
    """Median BOA per patch (percent) of the chosen models in each range of class share."""
    table = evaluation.boa_by_share(combined)
    wide = (table["mean"] * 100).unstack("model")[list(models)].rename(columns=LABELS)
    wide.insert(0, "patches", table["n"].groupby(level=["experiment", "bin"]).first())
    wide.index.names = ["experiment", "class share"]
    return wide


def figure_by_share(combined, experiment_name: str, models: Sequence[str] = SHARE_MODELS) -> Figure:
    """BOA of the chosen models per range of the experiment's class share."""
    table = evaluation.boa_by_share(combined).loc[experiment_name]
    return plots.plot_boa_by_share(
        table, list(models), LABELS,
        f"{experiment_name}: BOA by share of {SHARE_CLASSES[experiment_name]} pixels in the "
        "patch, L1C test, R-G-NIR and 13 bands",
        colors=COLORS,
    )


def figure_name(name: str) -> str:
    """Name for phase2_eval.show, kept apart from the phase 2 figures."""
    return f"{EXPERIMENT.name}-{name}"
