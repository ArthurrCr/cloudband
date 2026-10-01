"""Provenance manifest for a training run, built on the shared Manifest record."""

from __future__ import annotations

from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version

from cloudband.provenance.manifest import Manifest, build_manifest
from cloudband.train.loop import FitResult
from cloudband.train.protocol import TrainProtocol

TRAIN_DATASET = "cloudsen12plus_train_p509_high"

TRACKED_PACKAGES = ("torch", "fastai", "timm", "transformers", "scipy", "numpy")


def package_versions(names: tuple = TRACKED_PACKAGES) -> dict:
    """Collect the installed versions of packages whose numerics affect training."""
    versions = {}
    for name in names:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            continue
    return versions


def protocol_config(
    protocol: TrainProtocol, sampled_gsds: tuple = (), valid_losses: tuple = ()
) -> dict:
    """Turn a protocol into a plain config dict, plus what the run produced.

    valid_losses is the validation loss after every epoch; with it the epoch of
    the best checkpoint is recorded, so convergence can be checked later.
    """
    config = asdict(protocol)
    config["sampled_gsds_m"] = list(sampled_gsds)
    config["valid_loss_history"] = [float(value) for value in valid_losses]
    config["best_epoch"] = (
        min(range(len(valid_losses)), key=lambda i: valid_losses[i])
        if valid_losses
        else None
    )
    return config


def build_training_manifest(result: FitResult, protocol: TrainProtocol) -> Manifest:
    """Assemble the provenance manifest for one completed training run."""
    return build_manifest(
        result_name=result.checkpoint_name,
        dataset=TRAIN_DATASET,
        model_id=result.checkpoint_name,
        seed=result.seed,
        package_versions=package_versions(),
        config=protocol_config(protocol, result.sampled_gsds, result.valid_losses),
    )