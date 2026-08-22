"""Inference wrapper around the OmniCloudMask package with a pinned configuration.

Inference carries no random component, so there is no seed to fix. What must be
pinned is the configuration below: patch size and overlap change the mosaicking
at patch borders, and dtype changes marginal pixels.

Keyword names are validated against the installed signature rather than assumed,
so an unsupported option fails immediately instead of being silently ignored.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

PINNED_VERSION = "1.7.0"

# Weight set the package ships. The default is the newest available, which is
# not the one the published paper values were produced with.
LATEST_MODEL_VERSION = 4.0
PAPER_MODEL_VERSION = 1.0

# The models take red, green and near-infrared, in that order.
EXPECTED_CHANNELS = 3


@dataclass(frozen=True)
class InferenceConfig:
    """Options passed to the package, recorded alongside every result."""

    patch_size: int = 1000
    patch_overlap: int = 300
    inference_dtype: str = "float32"
    batch_size: int = 1
    model_version: float = LATEST_MODEL_VERSION
    inference_device: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "patch_size": self.patch_size,
            "patch_overlap": self.patch_overlap,
            "inference_dtype": self.inference_dtype,
            "batch_size": self.batch_size,
            "model_version": self.model_version,
        }
        if self.inference_device is not None:
            kwargs["inference_device"] = self.inference_device
        kwargs.update(self.extra)
        return kwargs


def package_version() -> str:
    import omnicloudmask

    version: str = omnicloudmask.__version__
    return version


def check_version(expected: str = PINNED_VERSION) -> None:
    """Raise when the installed package differs from the pinned version."""
    installed = package_version()
    if installed != expected:
        raise RuntimeError(
            f"omnicloudmask {installed} installed but {expected} pinned; "
            "reported accuracy values are version specific"
        )


def validate_kwargs(function: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return kwargs unchanged, raising when one is absent from the signature."""
    parameters = inspect.signature(function).parameters
    accepts_any = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    if accepts_any:
        return kwargs
    unsupported = sorted(set(kwargs) - set(parameters))
    if unsupported:
        raise TypeError(
            f"{function.__name__} does not accept {unsupported}; "
            f"available keywords: {sorted(parameters)}"
        )
    return kwargs


def prediction_directory(root: Path, version: str | None = None) -> Path:
    """Return the version scoped directory predictions are written to."""
    return root / f"ocm_preds_v{version or package_version()}"


def predict_array(
    stack: NDArray[np.floating] | NDArray[np.integer],
    config: InferenceConfig | None = None,
) -> NDArray[np.integer]:
    """Run inference on one in-memory red, green, NIR stack.

    The stack must hold exactly three channels in that order. The returned mask
    carries the four class codes the package emits.
    """
    from omnicloudmask import predict_from_array

    if stack.ndim != 3 or stack.shape[0] != EXPECTED_CHANNELS:
        raise ValueError(
            f"expected a ({EXPECTED_CHANNELS}, height, width) red-green-NIR stack, "
            f"got {stack.shape}"
        )

    settings = config or InferenceConfig()
    kwargs = validate_kwargs(predict_from_array, settings.as_kwargs())
    mask = predict_from_array(stack, **kwargs)
    return np.asarray(mask).squeeze()


def predict_scenes(
    scene_paths: list[Path],
    output_dir: Path,
    config: InferenceConfig | None = None,
) -> list[Path]:
    """Run inference over Sentinel-2 SAFE directories and return mask paths."""
    from omnicloudmask import load_s2, predict_from_load_func

    settings = config or InferenceConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    kwargs = validate_kwargs(predict_from_load_func, settings.as_kwargs())
    paths: list[Path] = predict_from_load_func(
        scene_paths=scene_paths,
        load_func=load_s2,
        output_dir=output_dir,
        **kwargs,
    )
    return [Path(path) for path in paths]