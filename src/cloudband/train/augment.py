"""The four OmniCloudMask augmentations of the paper (Wright et al. 2025, section 2.9).

Ported from the authors' implementation (training/augs.py in the OmniCloudMask
repository): RandomRectangle, SceneEdge, BatchTear, BatchRot90 and BatchFlip, with the
parameters their training notebook passes. That repository is at a later model version
and also applies sharpen/blur, clipping of extremes, random crops and a wider resampling;
none of those are in the paper, so none are here.

Two places, because two of them act on reflectance and the others on geometry:

- per sample, on the raw values, before the dynamic Z-score (`augment_sample`): the
  random rectangle fills a region of some channels with a reflectance value, and the
  scene edge marks everything past a random line as no data, which the normalization
  then excludes from its statistics and sets to zero, as the authors' pipeline does.
  The label is left unchanged by both, on purpose (paper, 2.9).
- per batch, on the GPU, after the resolution is drawn (`GeometricAugmentation`): the
  rotation, the flips and the tear move image and label together, the same way for
  every sample of the batch, as in the authors' code.

Randomness comes from torch's generator: in a DataLoader worker it is seeded per worker
from the main process, and in the main process it is part of the state saved after
every epoch, so a seed fixes the augmentations and a resumed run continues them.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from fastai.callback.core import Callback
from numpy.typing import NDArray

from cloudband.train.protocol import AugmentationConfig


def _uniform(low: float, high: float) -> float:
    return low + (high - low) * torch.rand(()).item()


def _randint(low: int, high: int) -> int:
    """An integer in [low, high], both ends included, like random.randint."""
    return int(torch.randint(low, high + 1, ()).item())


def random_rectangle(image: NDArray[np.floating], config: AugmentationConfig) -> None:
    """Fill one random rectangle of a random subset of channels with one value, in place."""
    channels, height, width = image.shape
    area = height * width * _uniform(config.rectangle_min_area, config.rectangle_max_area)
    log_low = math.log(config.rectangle_min_aspect)
    aspect = math.exp(_uniform(log_low, -log_low))

    def span(extent: float, size: int) -> slice:
        bound = int(round(math.sqrt(extent)))
        start = _randint(0, max(size - bound, 0))
        return slice(start, start + bound)

    rows, cols = span(area * aspect, height), span(area / aspect, width)
    count = _randint(1, channels)
    chosen = torch.randperm(channels)[:count].tolist()
    value = _randint(0, config.rectangle_max_fill)
    for channel in chosen:
        image[channel, rows, cols] = value


def scene_edge(image: NDArray[np.floating], nodata_value: float) -> None:
    """Mark everything on one side of a random line as no data, in place."""
    _, height, width = image.shape
    x0, y0 = _uniform(0, width), _uniform(0, height)
    angle = _uniform(0, 2 * math.pi)
    a, b = -math.sin(angle), math.cos(angle)
    c = -(a * x0 + b * y0)
    cols = np.arange(width)[np.newaxis, :]
    rows = np.arange(height)[:, np.newaxis]
    image[:, a * cols + b * rows + c > 0] = nodata_value


def augment_sample(
    image: NDArray[np.floating], config: AugmentationConfig, nodata_value: float
) -> NDArray[np.floating]:
    """The reflectance augmentations of one training sample, on a copy of its raw values."""
    image = image.copy()
    if torch.rand(()).item() < config.random_rectangle_prob:
        random_rectangle(image, config)
    if torch.rand(()).item() < config.scene_edge_prob:
        scene_edge(image, nodata_value)
    return image


def tear(
    tensor: torch.Tensor, angle_deg: float, x0: float, y0: float, displacement: int
) -> torch.Tensor:
    """Shift the part of the image far from a random line, as BatchTear does.

    Works on a batch of images (B, C, H, W) or of labels (B, H, W) with the same
    arguments, so both move together.
    """
    crop = int(displacement)
    if 2 * crop >= min(tensor.shape[-2:]):
        return tensor
    labels = tensor.dim() == 3
    work = tensor.unsqueeze(1) if labels else tensor
    dtype = work.dtype
    work = work.float()
    _, _, height, width = work.shape

    angle = math.radians(angle_deg)
    dx, dy = math.cos(angle), math.sin(angle)
    xx, yy = torch.meshgrid(
        torch.arange(width, device=work.device),
        torch.arange(height, device=work.device),
        indexing="xy",
    )
    distance = torch.abs(dx * (yy - y0) - dy * (xx - x0))
    distance = torch.clamp(distance / distance.max() * displacement, 0, displacement)
    shifted = torch.roll(
        work, shifts=(int(dy * displacement), int(dx * displacement)), dims=(2, 3)
    )
    torn = torch.where(distance > displacement / 2, shifted, work)
    torn = torn[:, :, crop:-crop, crop:-crop]
    torn = F.interpolate(torn, size=(height, width), mode="nearest").to(dtype)
    return torn.squeeze(1) if labels else torn


class GeometricAugmentation(Callback):
    """Rotation, flips and tear of every training batch, image and label alike.

    Runs after the resolution of the batch is drawn (MixedResolutionCallback, order
    0), as in the authors' pipeline, where resampling comes first. Validation batches
    are left as they are.
    """

    order = 5

    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.last: dict = {}

    def before_batch(self) -> None:
        if not self.training:
            return
        images, labels = self.learn.xb[0], self.learn.yb[0]
        applied = {"rotation": 0, "flip_h": False, "flip_v": False, "tear": False}

        if torch.rand(()).item() < self.config.rotation_flip_prob:
            turns = _randint(0, 3)
            images = torch.rot90(images, turns, dims=(-2, -1))
            labels = torch.rot90(labels, turns, dims=(-2, -1))
            applied["rotation"] = turns
            if torch.rand(()).item() < 0.5:
                images, labels = images.flip(-1), labels.flip(-1)
                applied["flip_h"] = True
            if torch.rand(()).item() < 0.5:
                images, labels = images.flip(-2), labels.flip(-2)
                applied["flip_v"] = True

        if torch.rand(()).item() < self.config.image_tear_prob:
            height, width = images.shape[-2:]
            angle = _uniform(0, 180)
            x0, y0 = _uniform(0, width), _uniform(0, height)
            images = tear(images, angle, x0, y0, self.config.tear_displacement)
            labels = tear(labels, angle, x0, y0, self.config.tear_displacement)
            applied["tear"] = True

        self.learn.xb = (images.contiguous(),)
        self.learn.yb = (labels.contiguous(),)
        self.last = applied
