import numpy as np
import torch
from fastai.data.core import DataLoaders

from cloudband.datasets.cloudsen12 import Sample
from cloudband.train.data import build_dataloaders
from cloudband.train.phase2 import full_protocol, run_phase2
from cloudband.train.protocol import LR_SEARCH_GRID, FULL_FROZEN_EPOCHS, TrainProtocol

PATCH_SIZE = 32


def fake_read_sample(table, index, crop_to_valid=True):
    rng = np.random.default_rng(index)
    image = rng.integers(0, 10000, size=(13, PATCH_SIZE, PATCH_SIZE))
    annotation = rng.integers(0, 4, size=(PATCH_SIZE, PATCH_SIZE))
    return Sample(
        identifier=f"scene-{index}",
        image=image.astype(np.int32),
        annotation=annotation.astype(np.int32),
    )


class TinyModelStandIn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=1)

    def forward(self, x):
        return self.conv(x.float())


def build_test_dls() -> DataLoaders:
    return build_dataloaders(
        train_table=range(4),
        valid_table=range(2),
        micro_batch_size=2,
        read_sample=fake_read_sample,
    )


def test_full_protocol_uses_the_winning_lr_and_the_full_epoch_budget():
    base = TrainProtocol(
        run_id="ocm-rgn-cs12-shared", architecture="ocm", learning_rate=1e-3
    )
    expanded = full_protocol(base, learning_rate=5e-4)

    assert expanded.learning_rate == 5e-4
    assert expanded.frozen_epochs == FULL_FROZEN_EPOCHS
    assert expanded.frozen_epochs != 0


def test_run_phase2_searches_once_and_trains_once_per_seed():
    dls = build_test_dls()
    protocol = TrainProtocol(
        run_id="swin-rgn-cs12-shared",
        architecture="swin",
        learning_rate=1e-3,
        effective_batch_size=2,
    )
    seeds = (0, 1, 2)

    results = run_phase2(dls, protocol, seeds=seeds, model_builder=TinyModelStandIn)

    assert len(results) == len(seeds)
    assert [r.fit_result.seed for r in results] == list(seeds)

    # the search ran once: every run shares the exact same search manifest
    # and the exact same winning learning rate, not a different one per seed
    assert len({id(r.lr_search) for r in results}) == 1
    winning_rates = {r.winning_protocol.learning_rate for r in results}
    assert len(winning_rates) == 1

    for result in results:
        assert len(result.lr_search.runs) == len(LR_SEARCH_GRID)
        assert result.winning_protocol.frozen_epochs != 0
        assert result.fit_result.best_val_loss < float("inf")
        assert result.manifest.seed == result.fit_result.seed


def test_run_phase2_defaults_the_search_seed_to_the_first_seed():
    dls = build_test_dls()
    protocol = TrainProtocol(
        run_id="swin-rgn-cs12-shared",
        architecture="swin",
        learning_rate=1e-3,
        effective_batch_size=2,
    )

    explicit = run_phase2(
        dls, protocol, seeds=(0, 1), model_builder=TinyModelStandIn, search_seed=0
    )
    default = run_phase2(dls, protocol, seeds=(0, 1), model_builder=TinyModelStandIn)

    explicit_rate = explicit[0].winning_protocol.learning_rate
    default_rate = default[0].winning_protocol.learning_rate
    assert explicit_rate == default_rate


def test_run_phase2_only_builds_one_model_per_search_candidate_plus_per_seed():
    dls = build_test_dls()
    protocol = TrainProtocol(
        run_id="swin-rgn-cs12-shared",
        architecture="swin",
        learning_rate=1e-3,
        effective_batch_size=2,
    )
    seeds = (0, 1, 2)
    seen_models = []

    def counting_builder():
        model = TinyModelStandIn()
        seen_models.append(model)
        return model

    run_phase2(dls, protocol, seeds=seeds, model_builder=counting_builder)

    expected = len(LR_SEARCH_GRID) + len(seeds)
    assert len(seen_models) == expected
    assert len({id(m) for m in seen_models}) == len(seen_models)