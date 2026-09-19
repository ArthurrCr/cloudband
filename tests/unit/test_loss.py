import torch

from cloudband.datasets.cloudsen12 import NODATA_VALUE
from cloudband.train.loss import build_loss


def test_build_loss_ignores_nodata_pixels():
    loss = build_loss()
    preds = torch.zeros(1, 4, 2, 2)
    preds[0, 0] = 10.0

    targets_all_correct = torch.zeros(1, 2, 2, dtype=torch.long)
    targets_with_nodata = targets_all_correct.clone()
    targets_with_nodata[0, 0, 0] = NODATA_VALUE

    loss_all_correct = loss(preds, targets_all_correct)
    loss_with_nodata = loss(preds, targets_with_nodata)

    assert torch.isclose(loss_all_correct, loss_with_nodata, atol=1e-5)


def test_build_loss_raises_without_ignore_index_on_the_same_batch():
    plain_loss = build_loss(ignore_index=-100)
    preds = torch.zeros(1, 4, 2, 2)
    targets = torch.zeros(1, 2, 2, dtype=torch.long)
    targets[0, 0, 0] = NODATA_VALUE

    try:
        plain_loss(preds, targets)
        raised = False
    except IndexError:
        raised = True

    assert raised