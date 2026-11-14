"""Baseline evaluation runs: predict, score, report and record provenance.

The notebook calls these functions and displays what they return. No logic
belongs in the notebook itself.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from itertools import islice
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cloudband.baselines import ocm
from cloudband.datasets import cloudsen12 as dataset
from cloudband.eval.confusion import BinaryConfusion
from cloudband.eval.experiments import EXPERIMENTS
from cloudband.eval.metrics import (
    balanced_overall_accuracy,
    producer_accuracy,
    user_accuracy,
)
from cloudband.eval.report import as_percentages, paper_style_table, to_frame
from cloudband.pipelines import cloudsen12 as scoring
from cloudband.provenance.manifest import build_manifest

# Zero-based channel positions of red, green and near-infrared in the L1C stack.
# The collection ships thirteen bands; OmniCloudMask takes these three.
RED_INDEX = 3
GREEN_INDEX = 2
NIR_INDEX = 8
RGN_INDICES: tuple[int, int, int] = (RED_INDEX, GREEN_INDEX, NIR_INDEX)
RGN_BANDS: tuple[str, str, str] = ("B04", "B03", "B8A")


@dataclass(frozen=True)
class Reference:
    """A published result, tied to the model and dataset that produced it.

    Both are required. A difference is a reproduction check only when the model
    and the dataset both match the run; otherwise it compares two different
    things and has to say so.
    """

    name: str
    model_id: str
    dataset_id: str
    boa: dict[str, float]
    note: str = ""

    def __post_init__(self) -> None:
        if not self.dataset_id:
            raise ValueError("a reference must name the dataset it was measured on")
        if not self.model_id:
            raise ValueError("a reference must name the model that produced it")
        unknown = sorted(set(self.boa) - set(EXPERIMENTS))
        if unknown:
            raise ValueError(f"unknown experiments in reference: {unknown}")


CLOUDSEN12_TEST = "cloudsen12plus_test_p509_high"
CLOUDSEN12_TEST_L2A = "cloudsen12plus_test_l2a_p509_high"
PIXBOX_S2 = "pixbox_s2"
PIXBOX_L8 = "pixbox_l8"

OCM_PAPER_MODEL = "ocm-rgn-published-v1.0"
OCM_CURRENT_MODEL = "ocm-rgn-published-v1.7.0"
STUPMASK_MODEL = "stup-7b-mcvm-native"

# Published results, each tied to the model and dataset that produced it.
PIXBOX_S2_PAPER = Reference(
    name="OmniCloudMask paper, Table A.1",
    model_id=OCM_PAPER_MODEL,
    dataset_id=PIXBOX_S2,
    boa={"clear": 92.2, "cloud": 91.2, "shadow": 80.5},
    note="model version 1.0, whose weights no longer load; see DIVERGENCES D13",
)

PIXBOX_S2_V1_7_0 = Reference(
    name="OmniCloudMask reference notebook, package 1.7.0",
    model_id=OCM_CURRENT_MODEL,
    dataset_id=PIXBOX_S2,
    boa={"clear": 92.42, "cloud": 91.52, "shadow": 81.37},
)

PIXBOX_L8_PAPER = Reference(
    name="OmniCloudMask paper, Table A.2",
    model_id=OCM_PAPER_MODEL,
    dataset_id=PIXBOX_L8,
    boa={"clear": 91.5, "cloud": 91.5, "shadow": 75.2},
)

STUPMASK_CLOUDSEN12 = Reference(
    name="STUPmask paper, section 4.4.1",
    model_id=STUPMASK_MODEL,
    dataset_id=CLOUDSEN12_TEST,
    boa={"cloud": 93.64},
    note="binary model on CloudSEN12, not CloudSEN12+; see DIVERGENCES D1, D2 and D5",
)


@dataclass(frozen=True)
class RunResult:
    """Everything one baseline run produces."""

    model_id: str
    dataset_id: str
    per_scene: dict[str, dict[str, BinaryConfusion]]
    pooled: dict[str, BinaryConfusion]
    scores: pd.DataFrame
    per_scene_boa: pd.DataFrame
    per_scene_pa: pd.DataFrame
    per_scene_ua: pd.DataFrame
    pairable: dict[str, int]


def select_rgn(stack: NDArray[np.integer]) -> NDArray[np.integer]:
    """Pull red, green and near-infrared out of the full band stack."""
    if stack.shape[0] <= max(RGN_INDICES):
        raise ValueError(
            f"stack has {stack.shape[0]} channels, need at least {max(RGN_INDICES) + 1}"
        )
    selected: NDArray[np.integer] = stack[list(RGN_INDICES)]
    return selected


def build_predictor(
    predict: Callable[[NDArray[np.integer]], NDArray[np.integer]],
) -> scoring.Predictor:
    """Wrap a three-channel predictor so it accepts a full band stack."""

    def predictor(stack: NDArray[np.integer]) -> NDArray[np.integer]:
        return predict(select_rgn(stack))

    return predictor


def load_test_split(source: str = dataset.REMOTE_L1C, verify: bool = True) -> pd.DataFrame:
    """Load the index and narrow it to the high quality p509 test split."""
    table = dataset.load_remote(source)
    if verify:
        differences = dataset.verify_split_sizes(table)
        if differences:
            raise ValueError(f"split sizes differ from the documented ones: {differences}")
    return dataset.select(table, split="test")


def run(
    table: pd.DataFrame,
    predict: Callable[[NDArray[np.integer]], NDArray[np.integer]],
    model_id: str,
    dataset_id: str = CLOUDSEN12_TEST,
    limit: int | None = None,
    progress: Callable[[int, int], None] | None = None,
    samples: Iterable[dataset.Sample] | None = None,
) -> RunResult:
    """Score every scene in the table and assemble the run result.

    The patches are read from the remote file, all thirteen bands, unless samples
    are given: those come from the local cache with red, green and NIR already
    selected, in the table's order.
    """
    predictor = predict if samples is not None else build_predictor(predict)
    source = (
        dataset.iter_samples(table, limit=limit) if samples is None else islice(samples, limit)
    )
    total = len(table) if limit is None else min(limit, len(table))

    def counted() -> Iterator[dataset.Sample]:
        for position, sample in enumerate(source, start=1):
            if progress is not None:
                progress(position, total)
            yield sample

    per_scene = scoring.score_split(counted(), predictor)
    pooled = scoring.pooled(per_scene)
    per_scene_boa = scoring.per_scene_metric(per_scene, balanced_overall_accuracy)
    per_scene_pa = scoring.per_scene_metric(per_scene, producer_accuracy)
    per_scene_ua = scoring.per_scene_metric(per_scene, user_accuracy)
    return RunResult(
        model_id=model_id,
        dataset_id=dataset_id,
        per_scene=per_scene,
        pooled=pooled,
        scores=as_percentages(to_frame(pooled)),
        per_scene_boa=per_scene_boa,
        per_scene_pa=per_scene_pa,
        per_scene_ua=per_scene_ua,
        pairable=scoring.pairable_scene_counts(per_scene_boa),
    )


def compare_to_reference(result: RunResult, reference: Reference) -> pd.DataFrame:
    """Put pooled BOA next to a published reference and show the difference.

    The frame carries whether the model and the dataset match the run, so a
    report cannot present a comparison between two models as a reproduction.
    """
    observed = {name: result.scores.loc[name, "boa"] for name in reference.boa}
    frame = pd.DataFrame(
        {
            "observed": pd.Series(observed),
            "reference": pd.Series(reference.boa),
        }
    )
    frame["difference"] = frame["observed"] - frame["reference"]
    frame["same_model"] = reference.model_id == result.model_id
    frame["same_dataset"] = reference.dataset_id == result.dataset_id
    frame.attrs["reference_name"] = reference.name
    frame.attrs["reference_model"] = reference.model_id
    frame.attrs["reference_dataset"] = reference.dataset_id
    frame.attrs["note"] = reference.note
    return frame


def describe_comparison(frame: pd.DataFrame) -> str:
    """State what the comparison in a frame measures."""
    name = frame.attrs.get("reference_name", "unknown reference")
    model = frame.attrs.get("reference_model", "unknown model")
    dataset_id = frame.attrs.get("reference_dataset", "unknown dataset")
    same_model = bool(frame["same_model"].all())
    same_dataset = bool(frame["same_dataset"].all())

    if same_model and same_dataset:
        return f"reproduction check against {name}"
    if same_dataset:
        return (
            f"comparison against a different model, {model}, on the same dataset; "
            "the difference reflects the models, not reproduction fidelity"
        )
    if same_model:
        return (
            f"comparison against {name} measured on {dataset_id}; "
            "the populations differ, so the difference is not a fidelity check"
        )
    return (
        f"comparison against {model} measured on {dataset_id}; "
        "neither the model nor the dataset matches this run"
    )


def save(
    result: RunResult,
    output_dir: Path,
    config: dict[str, object] | None = None,
    package_versions: dict[str, str] | None = None,
) -> dict[str, Path]:
    """Write scores, per-scene values and the manifest, and return their paths."""
    directory = Path(output_dir) / result.model_id / result.dataset_id
    directory.mkdir(parents=True, exist_ok=True)

    counts = pd.DataFrame(
        {
            name: {
                "tp": confusion.tp,
                "tn": confusion.tn,
                "fp": confusion.fp,
                "fn": confusion.fn,
            }
            for name, confusion in result.pooled.items()
        }
    ).transpose()

    paths = {
        "scores": directory / "scores.csv",
        "confusion": directory / "confusion.csv",
        "per_scene_boa": directory / "per_scene_boa.csv",
    }
    result.scores.to_csv(paths["scores"])
    counts.to_csv(paths["confusion"])
    result.per_scene_boa.to_csv(paths["per_scene_boa"])

    manifest = build_manifest(
        result_name=f"{result.model_id}__{result.dataset_id}",
        dataset=result.dataset_id,
        model_id=result.model_id,
        package_versions=package_versions or {},
        config={
            **(config or {}),
            "bands": list(RGN_BANDS),
            "band_indices": list(RGN_INDICES),
            "scenes": len(result.per_scene),
            "pairable_scenes": result.pairable,
        },
    )
    paths["manifest"] = manifest.write(directory / "manifest.json")
    return paths

PUBLISHED_PATCH = {"patch_size": 509, "patch_overlap": 0}


def published_model_id(model_version: float) -> str:
    """Identifier of one published weight set run through the pinned package."""
    return f"ocm-rgn-published-v{ocm.package_version()}-model{model_version}"


def published_config(model_version: float) -> ocm.InferenceConfig:
    """One 509 pixel patch per CloudSEN12+ sample, so no mosaicking is involved."""
    return ocm.InferenceConfig(model_version=model_version, **PUBLISHED_PATCH)


def version_label(model_version: float) -> str:
    return f"V{int(model_version)}"


def progress_printer(every: int = 25) -> Callable[[int, int], None]:
    def progress(position: int, total: int) -> None:
        if position % every == 0 or position == total:
            print(f"{position}/{total}", flush=True)

    return progress


def _cached_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path, index_col="experiment") if path.exists() else None


def score_published_versions(
    table: pd.DataFrame,
    versions: tuple[float, ...],
    results_dir: Path,
    package_versions: dict[str, str] | None = None,
    limit: int | None = None,
    progress: Callable[[int, int], None] | None = None,
    log: Callable[[str], None] = print,
    dataset_id: str = CLOUDSEN12_TEST,
    samples: Callable[[], Iterable[dataset.Sample]] | None = None,
) -> tuple[dict[float, pd.DataFrame], dict[float, pd.DataFrame]]:
    """Pooled scores and the paper-style table of each published weight set.

    A version already saved under results_dir is loaded instead of run again:
    inference on fixed weights with a pinned configuration has no random part, so
    the saved run is what a new one would give. samples, when given, returns a fresh
    iterator of cached patches (bands selected) for each version run.
    """
    scores: dict[float, pd.DataFrame] = {}
    paper_tables: dict[float, pd.DataFrame] = {}
    for version in versions:
        model_id = published_model_id(version)
        directory = Path(results_dir) / model_id / dataset_id
        saved = _cached_csv(directory / "scores.csv"), _cached_csv(directory / "paper_table.csv")
        if saved[0] is not None and saved[1] is not None:
            log(f"model_version={version} -> loaded from cache: {model_id}")
            scores[version], paper_tables[version] = saved
            continue

        config = published_config(version)
        log(f"model_version={version} -> running: {model_id}")
        result = run(
            table,
            lambda stack, config=config: ocm.predict_array(stack, config),
            model_id=model_id,
            dataset_id=dataset_id,
            limit=limit,
            progress=progress,
            samples=None if samples is None else samples(),
        )
        log(f"  pairable scenes: {result.pairable}")
        log("  " + describe_comparison(compare_to_reference(result, STUPMASK_CLOUDSEN12)))
        for name, path in save(
            result, results_dir, config=config.as_kwargs(), package_versions=package_versions
        ).items():
            log(f"  saved {name}: {path}")

        paper_table = paper_style_table(
            result.per_scene_boa, result.per_scene_pa, result.per_scene_ua, result.pairable
        )
        paper_table.to_csv(directory / "paper_table.csv")
        log(f"  saved paper_table: {directory / 'paper_table.csv'}")
        scores[version], paper_tables[version] = result.scores, paper_table
    return scores, paper_tables
