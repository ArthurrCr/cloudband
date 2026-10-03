from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cloudband.datasets.cloudsen12 import Sample
from cloudband.train import smoke
from cloudband.train.data import build_dataloaders
from cloudband.train.protocol import ocm_shared_protocol, patch_size_px
from cloudband.train.smoke import (
    _fits_in_memory,
    _is_memory_error,
    loader_samples_per_second,
    probe_micro_batch_size,
)

PROTOCOL = ocm_shared_protocol(1e-3)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def test_the_largest_candidate_that_fits_is_chosen():
    fits = lambda batch: batch <= 4  # noqa: E731

    assert probe_micro_batch_size(Tiny, PROTOCOL, (8, 4, 2), fits=fits) == 4


def test_the_first_candidate_wins_when_everything_fits():
    assert probe_micro_batch_size(Tiny, PROTOCOL, (8, 4, 2), fits=lambda b: True) == 8


def test_nothing_fitting_is_an_error_that_says_where():
    with pytest.raises(RuntimeError, match="micro-batch of 2 .* 565 px"):
        probe_micro_batch_size(Tiny, PROTOCOL, (8, 4, 2), fits=lambda b: False)


def test_the_probe_tries_candidates_from_the_largest_down():
    tried = []

    def fits(batch):
        tried.append(batch)
        return False

    with pytest.raises(RuntimeError):
        probe_micro_batch_size(Tiny, PROTOCOL, (8, 4, 2), fits=fits)

    assert tried == [8, 4, 2]


def test_without_a_gpu_the_first_candidate_is_returned(monkeypatch):
    no_gpu = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setattr(smoke, "torch", no_gpu)

    assert probe_micro_batch_size(Tiny, PROTOCOL, (8, 4, 2)) == 8


def test_the_worst_case_size_is_the_one_at_the_smallest_distance():
    assert patch_size_px(PROTOCOL.min_gsd_m) == 565


def test_both_ways_cuda_reports_a_full_gpu_are_recognised():
    assert _is_memory_error(torch.cuda.OutOfMemoryError("CUDA out of memory"))
    assert _is_memory_error(
        RuntimeError("cuDNN: CUDNN_STATUS_INTERNAL_ERROR_DEVICE_ALLOCATION_FAILED")
    )
    assert not _is_memory_error(RuntimeError("shape mismatch"))
    assert not _is_memory_error(ValueError("out of memory"))


def test_the_training_step_of_the_probe_runs_end_to_end():
    # same code path as on the GPU, on the CPU, with a tiny size
    assert _fits_in_memory(Tiny, 2, 16, False, 0.85, device="cpu") is True


class Greedy(Tiny):
    """Raises the error CUDA raises when a batch is too big for the GPU."""

    def forward(self, x):
        if x.shape[0] > 2:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate")
        return super().forward(x)


def test_a_model_that_runs_out_of_memory_is_reported_as_not_fitting():
    assert _fits_in_memory(Greedy, 4, 16, False, 0.85, device="cpu") is False
    assert _fits_in_memory(Greedy, 2, 16, False, 0.85, device="cpu") is True


def test_any_other_error_is_not_hidden():
    class Broken(Tiny):
        def forward(self, x):
            raise RuntimeError("shape mismatch")

    with pytest.raises(RuntimeError, match="shape mismatch"):
        _fits_in_memory(Broken, 2, 16, False, 0.85, device="cpu")


class TorchClaimingAGpu:
    """What the smoke module sees as torch: the real one, but with a GPU present.

    Only the module's own view is replaced; patching torch.cuda itself would make
    PyTorch's internals try to start CUDA, which does not exist here.
    """

    cuda = SimpleNamespace(
        is_available=lambda: True, OutOfMemoryError=torch.cuda.OutOfMemoryError
    )

    def __getattr__(self, name):
        return getattr(torch, name)


def test_the_probe_can_pick_a_smaller_batch_through_the_real_step(monkeypatch):
    # route the probe's measurement through the real step, on the CPU
    monkeypatch.setattr(smoke, "torch", TorchClaimingAGpu())
    monkeypatch.setattr(
        smoke,
        "_fits_in_memory",
        lambda builder, batch, size, amp, headroom: _fits_in_memory(
            builder, batch, 16, amp, headroom, device="cpu"
        ),
    )

    assert probe_micro_batch_size(Greedy, PROTOCOL, (8, 4, 2)) == 2


def fake_read_sample(table, index, crop_to_valid=True):
    rng = np.random.default_rng(index)
    return Sample(
        identifier=f"scene-{index}",
        image=rng.integers(0, 10000, (13, 32, 32)).astype(np.int32),
        annotation=rng.integers(0, 4, (32, 32)).astype(np.int32),
    )


def test_loader_throughput_is_a_positive_rate_in_samples_per_second():
    dls = build_dataloaders(
        train_table=range(40), valid_table=range(4), micro_batch_size=4,
        read_sample=fake_read_sample,
    )

    rate = loader_samples_per_second(dls, max_batches=5)

    assert rate > 0