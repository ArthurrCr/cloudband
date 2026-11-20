"""Swin+UPerNet over R-G-NIR or any band set, built from Hugging Face transformers."""

from __future__ import annotations

import torch
from fastai.data.core import DataLoaders
from fastai.learner import Learner
from transformers import (
    AutoBackbone,
    SwinConfig,
    UperNetConfig,
    UperNetForSemanticSegmentation,
)

from cloudband.datasets.cloudsen12 import VALID_SIZE
from cloudband.pipelines.phase0 import RGN_BANDS
from cloudband.train.loop import fit_protocol
from cloudband.train.loss import build_loss
from cloudband.train.phase2 import run_phase2, run_phase2_resumable
from cloudband.train.protocol import TrainProtocol
from cloudband.train.store import RunStore

INPUT_CHANNELS = len(RGN_BANDS)
OUTPUT_CLASSES = 4

SWIN_DEPTHS = (2, 2, 6, 2)
SWIN_WINDOW_SIZE = 7
# Indices into the backbone's stage_names, ["stem", "stage1", ..., "stage4"]: index 0
# is the stem (the patch embedding, which has the same stride as stage1), so the four
# Swin stages the UPerNet decoder is built for are 1 to 4. Using 0 to 3 instead drops
# the deepest stage and feeds the decoder the stem twice.
SWIN_ENCODER_STAGES = (1, 2, 3, 4)
SWIN_STAGE_CHANNELS = (96, 192, 384, 768)
SWIN_STAGE_STRIDES = (4, 8, 16, 32)
UPERNET_POOL_SCALES = (1, 2, 3, 6)

PRETRAINED_BACKBONE_ID = "microsoft/swin-tiny-patch4-window7-224"


class SwinUperNet(torch.nn.Module):
    """Wraps the HF model so forward returns a plain logits tensor."""

    def __init__(self, model: UperNetForSemanticSegmentation):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=x).logits


def adapt_patch_embedding(backbone, in_chans: int) -> None:
    """Give a pretrained Swin backbone an input of in_chans bands, in place.

    The ImageNet patch embedding takes three channels. It is replaced by one taking
    in_chans, whose weights follow the rule timm applies to the OCM encoders
    (timm.models.adapt_input_conv: the RGB filters repeated over the channels and
    scaled by 3 / in_chans), so both architectures start from the same adaptation.
    """
    from timm.models import adapt_input_conv

    # The module sits at backbone.embeddings in transformers 4 and at
    # backbone.swin.embeddings in transformers 5, so it is found by its class.
    embeddings = next(
        module for module in backbone.modules()
        if type(module).__name__ == "SwinPatchEmbeddings"
    )
    old = embeddings.projection
    new = torch.nn.Conv2d(
        in_chans, old.out_channels, kernel_size=old.kernel_size, stride=old.stride,
        padding=old.padding, bias=old.bias is not None,
    )
    with torch.no_grad():
        new.weight.copy_(adapt_input_conv(in_chans, old.weight))
        if old.bias is not None:
            new.bias.copy_(old.bias)
    embeddings.projection = new
    embeddings.num_channels = in_chans
    backbone.config.num_channels = in_chans


def build_swin_upernet(
    img_size: int = VALID_SIZE, pretrained: bool = True, in_chans: int = INPUT_CHANNELS
) -> torch.nn.Module:
    """Build Swin-T + UPerNet over in_chans input bands (R-G-NIR by default).

    The auxiliary head is disabled so the training loss stays plain cross
    entropy, the same shape as OCM's loss. pretrained loads only the Swin
    encoder's ImageNet-1k classification weights; the UPerNet decoder always
    starts from random initialisation, the same split OCM's own U-Net
    decoder uses between its pretrained encoder and its random decoder.
    """
    swin_config = SwinConfig(
        num_channels=in_chans,
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
        if in_chans != model.backbone.config.num_channels:
            adapt_patch_embedding(model.backbone, in_chans)

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
    seeds: tuple,
    img_size: int = VALID_SIZE,
    pretrained: bool = True,
    search_seed: int | None = None,
) -> tuple:
    """Search the learning rate once, then train Swin+UPerNet once per seed."""
    return run_phase2(
        dls,
        protocol,
        seeds,
        model_builder=lambda: build_swin_upernet(
            img_size=img_size, pretrained=pretrained
        ),
        search_seed=search_seed,
    )


def run_swin_upernet_resumable(
    dls: DataLoaders,
    protocol: TrainProtocol,
    seeds: tuple,
    store: RunStore,
    img_size: int = VALID_SIZE,
    pretrained: bool = True,
    search_seed: int | None = None,
    progress=print,
    accept_edge_winner: bool = False,
    break_locks: bool = False,
) -> dict[str, str]:
    """Like run_swin_upernet_phase2, but saving each run and skipping saved ones."""
    return run_phase2_resumable(
        dls,
        protocol,
        seeds,
        model_builder=lambda: build_swin_upernet(
            img_size=img_size, pretrained=pretrained
        ),
        store=store,
        search_seed=search_seed,
        progress=progress,
        accept_edge_winner=accept_edge_winner,
        break_locks=break_locks,
    )