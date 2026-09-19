import torch

from cloudband.models.ocm import OcmEnsemble, PAPER_BACKBONES, build_unet

NATIVE_SIZE = 509
MIXED_RES_SIZE = 231


def test_build_unet_output_matches_input_spatial_size_for_every_backbone():
    for backbone_name in PAPER_BACKBONES:
        model = build_unet(
            backbone_name, img_size=(NATIVE_SIZE, NATIVE_SIZE), pretrained=False
        )
        native_input = torch.randn(1, 3, NATIVE_SIZE, NATIVE_SIZE)
        assert model(native_input).shape == (1, 4, NATIVE_SIZE, NATIVE_SIZE)


def test_build_unet_handles_a_mixed_resolution_size_not_used_at_construction():
    for backbone_name in PAPER_BACKBONES:
        model = build_unet(
            backbone_name, img_size=(NATIVE_SIZE, NATIVE_SIZE), pretrained=False
        )
        resized_input = torch.randn(1, 3, MIXED_RES_SIZE, MIXED_RES_SIZE)
        output = model(resized_input)
        assert output.shape == (1, 4, MIXED_RES_SIZE, MIXED_RES_SIZE)


class ConstantLogits(torch.nn.Module):
    """A stand-in model that always predicts fixed, known logits."""

    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.logits = logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        return self.logits.expand(batch, *self.logits.shape[1:])


def test_ocm_ensemble_averages_softmax_probabilities_not_logits():
    logits_a = torch.tensor([[[[10.0]], [[0.0]], [[0.0]], [[0.0]]]])
    logits_b = torch.tensor([[[[0.0]], [[10.0]], [[0.0]], [[0.0]]]])
    ensemble = OcmEnsemble((ConstantLogits(logits_a), ConstantLogits(logits_b)))

    output = ensemble(torch.zeros(1, 3, 1, 1))

    expected = (torch.softmax(logits_a, dim=1) + torch.softmax(logits_b, dim=1)) / 2
    assert torch.allclose(output, expected, atol=1e-6)
    assert torch.allclose(output.sum(dim=1), torch.ones(1, 1, 1), atol=1e-6)


def test_ocm_ensemble_moves_input_to_its_own_device():
    class TinyParameterizedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, kernel_size=1)

        def forward(self, x):
            return self.conv(x)

    ensemble = OcmEnsemble((TinyParameterizedModel(), TinyParameterizedModel()))
    calls = []
    original_to = torch.Tensor.to

    def spy_to(self, *args, **kwargs):
        calls.append((args, kwargs))
        return original_to(self, *args, **kwargs)

    torch.Tensor.to = spy_to
    try:
        ensemble(torch.rand(1, 3, 4, 4))
    finally:
        torch.Tensor.to = original_to

    assert len(calls) >= 1


def test_ocm_ensemble_with_no_parameters_falls_back_to_the_input_device():
    ensemble = OcmEnsemble(
        (
            ConstantLogits(torch.zeros(1, 4, 1, 1)),
            ConstantLogits(torch.zeros(1, 4, 1, 1)),
        )
    )
    output = ensemble(torch.rand(1, 3, 1, 1))
    assert output.shape == (1, 4, 1, 1)