"""Swin+UPerNet, restricted to R-G-NIR, built from Hugging Face transformers."""

from __future__ import annotations

import torch
from fastai.data.core import DataLoaders
from fastai.learner import Learner
from transformers import AutoBackbone, SwinConfig, UperNetConfig
from transformers import UperNetForSemanticSegmentation

from cloudband.datasets.cloudsen12 import VALID_SIZE
from cloudband.pipelines.phase0 import RGN_BANDS
from cloudband.train.loop import fit_protocol
from cloudband.train.loss import build_loss
from cloudband.train.phase2 import Phase2Run, run_phase2
from cloudband.train.protocol import TrainProtocol

INPUT_CHANNELS = len(RGN_BANDS)
OUTPUT_CLASSES = 4

SWIN_DEPTHS = (2, 2, 6, 2)
SWIN_WINDOW_SIZE = 7
SWIN_ENCODER_STAGES = (0, 1, 2, 3)
UPERNET_POOL_SCALES = (1, 2, 3, 6)

PRETRAINED_BACKBONE_ID = "microsoft/swin-tiny-patch4-window7-224"


class SwinUperNet(torch.nn.Module):
    """Wraps the HF model so forward returns a plain logits tensor."""

    def __init__(self, model: UperNetForSemanticSegmentation):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=x).logits


def build_swin_upernet(
    img_size: int = VALID_SIZE, pretrained: bool = True
) -> torch.nn.Module:
    """Build Swin-T + UPerNet, restricted to R-G-NIR input.

    The auxiliary head is disabled so the training loss stays plain cross
    entropy, the same shape as OCM's loss. pretrained loads only the Swin
    encoder's ImageNet-1k classification weights; the UPerNet decoder always
    starts from random initialisation, the same split OCM's own U-Net
    decoder uses between its pretrained encoder and its random decoder.
    """
    swin_config = SwinConfig(
        num_channels=INPUT_CHANNELS,
        image_size=img_size,
        depths=SWIN_DEPTHS,
        window_size=SWIN_WINDOW_SIZE,
        out_indices=list(SWIN_ENCODER_STAGES),
    )
    config = UperNetConfig(
        backbone_config=swin_config,
        pool_scales=UPERNET_POOL_SCALES,
        use_auxiliary_head=False,
        num_labels=OUTPUT_CLASSES,
    )
    model = UperNetForSemanticSegmentation(config)

    if pretrained:
        model.backbone = AutoBackbone.from_pretrained(
            PRETRAINED_BACKBONE_ID,
            out_indices=list(SWIN_ENCODER_STAGES),
        )

    return SwinUperNet(model)


def train_swin_upernet(
    dls: DataLoaders,
    protocol: TrainProtocol,
    seed: int,
    img_size: int = VALID_SIZE,
    pretrained: bool = True,
    model_builder=build_swin_upernet,
):
    """Train the Swin+UPerNet model under the shared protocol."""
    model = model_builder(img_size=img_size, pretrained=pretrained)
    learner = Learner(dls, model, loss_func=build_loss())
    return fit_protocol(learner, protocol, seed=seed)


def run_swin_upernet_phase2(
    dls: DataLoaders,
    protocol: TrainProtocol,
    seed: int,
    img_size: int = VALID_SIZE,
    pretrained: bool = True,
) -> Phase2Run:
    """Search the learning rate, then train Swin+UPerNet at the full budget."""
    return run_phase2(
        dls,
        protocol,
        seed,
        model_builder=lambda: build_swin_upernet(
            img_size=img_size, pretrained=pretrained
        ),
    )