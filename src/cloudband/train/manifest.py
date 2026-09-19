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


def protocol_config(protocol: TrainProtocol, sampled_gsds: tuple = ()) -> dict:
    """Turn a protocol into a plain config dict, plus the GSDs actually sampled."""
    config = asdict(protocol)
    config["sampled_gsds_m"] = list(sampled_gsds)
    return config


def build_training_manifest(result: FitResult, protocol: TrainProtocol) -> Manifest:
    """Assemble the provenance manifest for one completed training run."""
    return build_manifest(
        result_name=result.checkpoint_name,
        dataset=TRAIN_DATASET,
        model_id=result.checkpoint_name,
        seed=result.seed,
        package_versions=package_versions(),
        config=protocol_config(protocol, result.sampled_gsds),
    )