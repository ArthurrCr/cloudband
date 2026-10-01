import torch

from cloudband.models.swin_upernet import (
    SWIN_STAGE_CHANNELS,
    SWIN_STAGE_STRIDES,
    build_swin_upernet,
)


def test_the_decoder_receives_the_four_swin_stages_not_the_stem():
    backbone = build_swin_upernet(pretrained=False).model.backbone

    # a stem in this list means index 0 was used and stage4 was dropped
    assert backbone.out_features == ["stage1", "stage2", "stage3", "stage4"]
    assert tuple(backbone.channels) == SWIN_STAGE_CHANNELS


def test_the_four_feature_maps_have_the_strides_swin_t_upernet_expects():
    backbone = build_swin_upernet(pretrained=False).model.backbone
    size = 224

    with torch.no_grad():
        features = backbone(pixel_values=torch.rand(1, 3, size, size)).feature_maps

    assert tuple(size // f.shape[-1] for f in features) == SWIN_STAGE_STRIDES
    assert tuple(f.shape[1] for f in features) == SWIN_STAGE_CHANNELS


def test_the_full_model_still_returns_four_class_logits_at_input_size():
    model = build_swin_upernet(pretrained=False).eval()

    with torch.no_grad():
        logits = model(torch.rand(2, 3, 224, 224))

    assert tuple(logits.shape) == (2, 4, 224, 224)