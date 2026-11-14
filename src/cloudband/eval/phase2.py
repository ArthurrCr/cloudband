"""Phase 2 test evaluation: confusion counts per patch, then every comparison from them.

The test patches are scored once per seed and processing level, for every model of the
comparison in the same pass: each OCM backbone on its own, the OCM ensemble (the mean of
the two backbones' softmax, exactly as OcmEnsemble), and Swin+UPerNet. What is kept is the
confusion matrix of every patch and experiment, not a metric, so the per-patch BOA, the
per-ROI BOA (the five patches of a location summed first) and the whole-split BOA all
come from one inference.

Why per ROI: the 975 test patches are 195 locations with five dates each, so patches of
one location are not independent. A paired test over patches counts each location five
times; over ROIs it counts it once. Both are reported, and every confidence interval
resamples ROIs, never single patches.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

from cloudband.datasets.cloudsen12 import Sample
from cloudband.eval.experiments import EXPERIMENTS, prediction_masks, run_experiments
from cloudband.labels import cloudsen12 as labels
from cloudband.train.normalize import dynamic_z_score

COUNTS = ("tp", "tn", "fp", "fn")
COLUMNS = ("model", "scene", "roi", "experiment", *COUNTS)


def score_models(
    samples: Iterable[Sample],
    members: Mapping[str, torch.nn.Module],
    ensembles: Mapping[str, tuple[str, ...]],
    device: str = "cpu",
) -> pd.DataFrame:
    """Confusion counts of every patch, for every member model and every ensemble.

    Samples hold the R-G-NIR bands, as the local cache returns them, and are normalized
    with the dynamic Z-score training uses. Each member runs once per patch; an
    ensemble averages its members' softmax and takes the arg max.
    """
    for name, parts in ensembles.items():
        missing = [part for part in parts if part not in members]
        if missing:
            raise ValueError(f"ensemble {name!r} needs models {missing}")
    for model in members.values():
        model.to(device).eval()

    rows = []
    for sample in samples:
        reference = labels.label_masks(sample.annotation)
        annotated = labels.valid_mask(sample.annotation)
        labels.verify_partition(reference, int(np.count_nonzero(annotated)))

        image = dynamic_z_score(sample.image.astype(np.float32))
        tensor = torch.from_numpy(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = {name: model(tensor) for name, model in members.items()}
            predicted = {name: values.argmax(dim=1) for name, values in logits.items()}
            for name, parts in ensembles.items():
                mean = torch.stack([torch.softmax(logits[p], dim=1) for p in parts]).mean(0)
                predicted[name] = mean.argmax(dim=1)

        for name, classes in predicted.items():
            prediction = classes.squeeze(0).cpu().numpy()
            rows.extend(_count_rows(name, sample, prediction, reference, annotated))
    if not rows:
        raise ValueError("no samples were scored")
    return pd.DataFrame(rows, columns=list(COLUMNS))


def score_predictor(
    samples: Iterable[Sample],
    predict: Callable[[np.ndarray], np.ndarray],
    name: str,
) -> pd.DataFrame:
    """Confusion counts of every patch for a model reached through a function.

    For the published OmniCloudMask, whose package takes the raw R-G-NIR stack and
    returns the class map, so it sees the same patches as the trained models, with
    its own normalization.
    """
    rows = []
    for sample in samples:
        reference = labels.label_masks(sample.annotation)
        annotated = labels.valid_mask(sample.annotation)
        labels.verify_partition(reference, int(np.count_nonzero(annotated)))
        prediction = np.asarray(predict(sample.image)).squeeze()
        rows.extend(_count_rows(name, sample, prediction, reference, annotated))
    if not rows:
        raise ValueError("no samples were scored")
    return pd.DataFrame(rows, columns=list(COLUMNS))


def _count_rows(name, sample, prediction, reference, annotated):
    prediction = prediction.astype(np.int64)
    if prediction.shape != sample.annotation.shape:
        raise ValueError(
            f"{name} predicted {prediction.shape} for an annotation of "
            f"{sample.annotation.shape}"
        )
    confusions = run_experiments(reference, prediction_masks(prediction), annotated)
    return [
        (name, sample.identifier, sample.roi_id, experiment,
         confusion.tp, confusion.tn, confusion.fp, confusion.fn)
        for experiment, confusion in confusions.items()
    ]


def with_reference(
    counts_by_seed: Mapping[int, pd.DataFrame], reference: pd.DataFrame
) -> dict[int, pd.DataFrame]:
    """Add a model without seeds (the published one) to every seed's counts.

    It must cover exactly the same patches, so a paired comparison never drops or
    invents one.
    """
    names = set(reference["model"])
    combined = {}
    for seed, frame in counts_by_seed.items():
        clash = names & set(frame["model"])
        if clash:
            raise ValueError(f"seed {seed} already has models {sorted(clash)}")
        if set(frame["scene"]) != set(reference["scene"]):
            raise ValueError(f"seed {seed} and the reference cover different patches")
        combined[seed] = pd.concat([frame, reference], ignore_index=True)
    return combined


def save_counts(frame: pd.DataFrame, path) -> None:
    """Write next to the target and rename, so a cut copy is never read as complete."""
    from pathlib import Path

    path = Path(path)
    partial = path.with_name(path.name + ".partial")
    frame.to_csv(partial, index=False)
    partial.replace(path)


def load_counts(path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"scene": str, "roi": str})
    missing = set(COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return frame


def boa(counts: pd.DataFrame) -> pd.Series:
    """Balanced overall accuracy from summed counts; NaN when a class side is absent."""
    positive = counts["tp"] + counts["fn"]
    negative = counts["tn"] + counts["fp"]
    sensitivity = counts["tp"] / positive.where(positive > 0)
    specificity = counts["tn"] / negative.where(negative > 0)
    return 0.5 * (sensitivity + specificity)


def unit_boa(counts: pd.DataFrame, unit: str) -> pd.DataFrame:
    """BOA per model and unit ("scene" or "roi"), one column per experiment.

    A ROI's counts are the sum of its patches' counts, so the ROI is scored as one
    image of five dates rather than as the mean of five scores.
    """
    if unit not in ("scene", "roi"):
        raise ValueError(f"unit must be 'scene' or 'roi', not {unit!r}")
    if unit == "roi" and counts["roi"].isna().any():
        raise ValueError("some patches have no ROI; the per-ROI analysis needs all of them")
    summed = counts.groupby(["model", unit, "experiment"])[list(COUNTS)].sum()
    return boa(summed).unstack("experiment")[list(EXPERIMENTS)]


def pooled_boa(counts: pd.DataFrame) -> pd.DataFrame:
    """BOA of the whole split per model: every patch's counts summed, as papers report."""
    summed = counts.groupby(["model", "experiment"])[list(COUNTS)].sum()
    return boa(summed).unstack("experiment")[list(EXPERIMENTS)]


def seed_table(counts_by_seed: Mapping[int, pd.DataFrame]) -> pd.DataFrame:
    """Whole-split BOA of every model and experiment: each seed, their mean and SD."""
    per_seed = {seed: pooled_boa(frame).stack() for seed, frame in counts_by_seed.items()}
    table = pd.DataFrame(per_seed)
    table.columns = [f"seed {seed}" for seed in table.columns]
    values = table.to_numpy()
    table.insert(0, "sd", values.std(axis=1, ddof=1) if values.shape[1] > 1 else np.nan)
    table.insert(0, "mean", values.mean(axis=1))
    table.index.names = ["model", "experiment"]
    return table


def seed_averaged(counts_by_seed: Mapping[int, pd.DataFrame], model: str, unit: str):
    """One model's BOA per unit, averaged across seeds."""
    frames = [unit_boa(frame, unit).loc[model] for frame in counts_by_seed.values()]
    return pd.concat(frames).groupby(level=0).mean()


def unit_differences(
    counts_by_seed: Mapping[int, pd.DataFrame], model_a: str, model_b: str, unit: str
) -> pd.DataFrame:
    """BOA of A minus BOA of B per unit, each averaged over seeds; NaN where undefined."""
    a = seed_averaged(counts_by_seed, model_a, unit)
    b = seed_averaged(counts_by_seed, model_b, unit)
    return a - b


def holm(p_values) -> np.ndarray:
    """Holm-Bonferroni adjustment, NaN left as NaN."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p, np.nan)
    present = np.flatnonzero(~np.isnan(p))
    order = present[np.argsort(p[present])]
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(order) - rank) * p[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def _cluster_bootstrap(differences: pd.Series, clusters: pd.Series, draws: int, seed: int):
    """95% interval of the mean difference, resampling whole ROIs with replacement."""
    grouped = pd.DataFrame({"d": differences.to_numpy(), "c": clusters.to_numpy()})
    by_cluster = grouped.groupby("c")["d"].agg(["sum", "count"])
    sums, sizes = by_cluster["sum"].to_numpy(), by_cluster["count"].to_numpy()
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(sums), size=(draws, len(sums)))
    means = sums[picks].sum(axis=1) / sizes[picks].sum(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_comparison(
    counts_by_seed: Mapping[int, pd.DataFrame],
    model_a: str,
    model_b: str,
    unit: str,
    draws: int = 10_000,
    seed: int = 0,
) -> pd.DataFrame:
    """Model A against model B, paired by unit, after averaging each model over seeds.

    Per experiment: units with a defined BOA for both, mean BOA of each, mean and median
    difference (A minus B), a 95% interval of the mean difference that resamples ROIs,
    the Wilcoxon signed-rank p-value, and that p-value adjusted with Holm over the three
    experiments.
    """
    a = seed_averaged(counts_by_seed, model_a, unit)
    b = seed_averaged(counts_by_seed, model_b, unit)
    any_frame = next(iter(counts_by_seed.values()))
    roi_of = any_frame.drop_duplicates(unit).set_index(unit)["roi"] if unit == "scene" else None

    rows = []
    for experiment in EXPERIMENTS:
        joined = pd.DataFrame({"a": a[experiment], "b": b[experiment]}).dropna()
        difference = joined["a"] - joined["b"]
        row = {"experiment": experiment, "n": len(joined)}
        if len(joined) == 0:
            rows.append(row)
            continue
        clusters = (
            pd.Series(joined.index, index=joined.index)
            if roi_of is None
            else roi_of.reindex(joined.index)
        )
        low, high = _cluster_bootstrap(difference, clusters, draws, seed)
        if (difference == 0).all():
            p_value = 1.0
        else:
            p_value = float(wilcoxon(joined["a"], joined["b"]).pvalue)
        row.update(
            {
                f"boa {model_a}": joined["a"].mean(),
                f"boa {model_b}": joined["b"].mean(),
                "mean diff": difference.mean(),
                "ci95 low": low,
                "ci95 high": high,
                "median diff": difference.median(),
                f"{model_a} better in": float((difference > 0).mean()),
                "p": p_value,
            }
        )
        rows.append(row)
    table = pd.DataFrame(rows).set_index("experiment")
    table["p holm"] = holm(table["p"]) if "p" in table else np.nan
    return table


def report(
    counts_by_seed: Mapping[int, pd.DataFrame], published: Sequence[str] = ()
) -> dict[str, pd.DataFrame]:
    """Every table of the comparison, from the counts of each seed.

    published names the published OmniCloudMask models in the counts. The OCM
    ensemble and Swin are each compared with every one of them. They were trained on
    other data and with another recipe, so they are reference points, not part of the
    controlled comparison.
    """
    if isinstance(published, str):
        published = (published,)
    tables = {
        "whole split, per seed": seed_table(counts_by_seed),
        "OCM vs Swin, per ROI": paired_comparison(counts_by_seed, "ocm", "swin", "roi"),
        "OCM vs Swin, per patch": paired_comparison(counts_by_seed, "ocm", "swin", "scene"),
        "regnety alone vs Swin, per ROI": paired_comparison(
            counts_by_seed, "regnety", "swin", "roi"
        ),
        "convnext alone vs Swin, per ROI": paired_comparison(
            counts_by_seed, "convnext", "swin", "roi"
        ),
    }
    for name in published:
        for model, label in (("ocm", "OCM"), ("swin", "Swin")):
            tables[f"{label} vs {name}, per ROI"] = paired_comparison(
                counts_by_seed, model, name, "roi"
            )
    return tables


def stacked(tables: Mapping[str, pd.DataFrame], titles: Sequence[str]) -> pd.DataFrame:
    """Several comparison tables in one, indexed by comparison and experiment.

    The per-model BOA columns are dropped, since their names differ between tables;
    what remains is the difference, its interval, the share of units and the p-values.
    """
    columns = ["n", "mean diff", "ci95 low", "ci95 high", "median diff", "p", "p holm"]
    frames = {}
    for title in titles:
        table = tables[title]
        better = next(c for c in table.columns if c.endswith(" better in"))
        frame = table[columns[:5]].assign(**{"first better in": table[better]})
        frames[title.split(",")[0]] = frame.join(table[columns[5:]])
    return pd.concat(frames, names=["comparison", "experiment"])


def legacy_check(counts: pd.DataFrame, model: str, legacy: pd.DataFrame) -> float:
    """Largest gap between the per-patch BOA from these counts and an earlier saved one."""
    fresh = unit_boa(counts, "scene").loc[model]
    legacy = legacy.copy()
    legacy.index = legacy.index.astype(str)
    joined = fresh.join(legacy, lsuffix="_new", rsuffix="_old", how="inner")
    gaps = [
        (joined[f"{e}_new"] - joined[f"{e}_old"]).abs().max()
        for e in EXPERIMENTS
        if f"{e}_old" in joined
    ]
    return float(np.nanmax(gaps)) if gaps else float("nan")
