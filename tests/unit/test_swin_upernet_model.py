import torch

from cloudband.models.swin_upernet import OUTPUT_CLASSES, build_swin_upernet

SIZES = (231, 339, 509, 565)


def test_output_matches_input_spatial_size_across_the_full_gsd_range():
    model = build_swin_upernet(img_size=509, pretrained=False)
    model.eval()
    for size in SIZES:
        dummy = torch.randn(1, 3, size, size)
        with torch.no_grad():
            output = model(dummy)
        assert output.shape == (1, OUTPUT_CLASSES, size, size)


def test_output_has_no_auxiliary_head_contribution():
    model = build_swin_upernet(img_size=509, pretrained=False)
    assert model.model.auxiliary_head is None


def test_backward_pass_runs_without_error():
    model = build_swin_upernet(img_size=509, pretrained=False)
    dummy = torch.randn(2, 3, 231, 231, requires_grad=False)
    target = torch.randint(0, OUTPUT_CLASSES, (2, 231, 231))

    output = model(dummy)
    loss = torch.nn.functional.cross_entropy(output, target)
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)


def test_training_mode_requires_batch_size_above_one():
    model = build_swin_upernet(img_size=509, pretrained=False)
    dummy = torch.randn(1, 3, 231, 231)

    try:
        model(dummy)
        raised = False
    except ValueError:
        raised = True

    assert raised