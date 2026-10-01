from functools import partial

import numpy as np
import pytest
import torch

from cloudband.datasets.cloudsen12 import Sample
from cloudband.models import ocm as ocm_module
from cloudband.train import phase2
from cloudband.train.data import build_dataloaders
from cloudband.train.loop import checkpoint_name
from cloudband.train.lr_search import search_learning_rate
from cloudband.train.protocol import LrSearchManifest, LrSearchRun, TrainProtocol
from cloudband.train.store import RunStore, load_model_from_checkpoint

PATCH_SIZE = 32
GRID = (1e-3, 5e-3)


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
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=1)

    def forward(self, x):
        return self.conv(x.float())


@pytest.fixture
def setup(tmp_path, monkeypatch):
    # fastai writes checkpoints under ./models, so run inside a scratch directory
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(phase2, "FULL_FROZEN_EPOCHS", 1)
    monkeypatch.setattr(phase2, "FULL_UNFROZEN_EPOCHS", 1)
    monkeypatch.setattr(
        phase2,
        "search_learning_rate",
        partial(search_learning_rate, frozen_epochs=1, unfrozen_epochs=1, grid=GRID),
    )
    dls = build_dataloaders(
        train_table=range(4),
        valid_table=range(2),
        micro_batch_size=2,
        read_sample=fake_read_sample,
    )
    store = RunStore(
        results_dir=tmp_path / "drive" / "results",
        checkpoint_dir=tmp_path / "drive" / "results" / "checkpoints",
        source_dir=tmp_path / "models",
    )
    protocol = TrainProtocol(
        run_id="swin-rgn-cs12-shared",
        architecture="swin",
        learning_rate=1e-3,
        effective_batch_size=2,
    )
    return dls, store, protocol


def test_a_run_is_done_only_with_both_checkpoint_and_manifest(tmp_path):
    store = RunStore(tmp_path / "r", tmp_path / "r" / "c", tmp_path / "m")

    assert not store.is_done("x")
    store.checkpoint_path("x").write_bytes(b"weights")
    assert not store.is_done("x")          # a checkpoint alone is not a finished run
    store.manifest_path("x").write_text("{}")
    assert store.is_done("x")


def test_a_half_copied_checkpoint_does_not_count_as_done(tmp_path):
    store = RunStore(tmp_path / "r", tmp_path / "r" / "c", tmp_path / "m")
    store.checkpoint_path("x").with_suffix(".pth.partial").write_bytes(b"half")

    assert not store.is_done("x")


def test_the_learning_rate_search_round_trips_through_the_store(tmp_path):
    store = RunStore(tmp_path / "r", tmp_path / "r" / "c", tmp_path / "m")
    protocol = TrainProtocol(run_id="a", architecture="ocm", learning_rate=1e-3)
    search = LrSearchManifest(
        architecture="ocm",
        runs=(LrSearchRun("a-lr1", 1e-3, 0.40), LrSearchRun("a-lr2", 5e-3, 0.30)),
        grid=GRID,
    )

    assert store.load_search(protocol) is None
    store.save_search(protocol, search)
    loaded = store.load_search(protocol)

    assert loaded == search
    assert loaded.winner().learning_rate == 5e-3


def test_saving_a_run_without_its_checkpoint_fails_loudly(setup):
    dls, store, protocol = setup
    lr_search, winning = phase2.search_phase2_learning_rate(
        dls, protocol, TinyModelStandIn, search_seed=0
    )
    run = phase2.train_phase2_run(dls, winning, lr_search, 0, TinyModelStandIn)
    (store.source_dir / f"{run.fit_result.checkpoint_name}.pth").unlink()

    with pytest.raises(FileNotFoundError):
        store.save_run(run)


def test_resuming_reuses_the_search_and_skips_finished_seeds(setup):
    dls, store, protocol = setup
    built = []

    def builder():
        built.append(1)
        return TinyModelStandIn()

    quiet = lambda _: None  # noqa: E731
    first = phase2.run_phase2_resumable(
        dls, protocol, (0,), builder, store, progress=quiet
    )
    after_first = len(built)
    assert list(first.values()) == ["trained"]
    assert after_first == len(GRID) + 1        # the search once, plus seed 0

    # the "session" dies and restarts with one more seed to do
    second = phase2.run_phase2_resumable(
        dls, protocol, (0, 1), builder, store, progress=quiet
    )

    assert sorted(second.values()) == ["skipped", "trained"]
    assert len(built) - after_first == 1        # no new search, only seed 1 trained
    for seed in (0, 1):
        assert store.is_done(checkpoint_name(_winning(store, protocol), seed))


def _winning(store, protocol):
    rate = store.load_search(protocol).winner().learning_rate
    return phase2.full_protocol(protocol, rate)


def test_a_finished_run_is_not_retrained_even_when_asked_again(setup):
    dls, store, protocol = setup
    phase2.run_phase2_resumable(
        dls, protocol, (0,), TinyModelStandIn, store, progress=lambda _: None
    )
    manifest = store.manifest_path(checkpoint_name(_winning(store, protocol), 0))
    before = manifest.stat().st_mtime_ns

    again = phase2.run_phase2_resumable(
        dls, protocol, (0,), TinyModelStandIn, store, progress=lambda _: None
    )

    assert list(again.values()) == ["skipped"]
    assert manifest.stat().st_mtime_ns == before


def test_ocm_ensemble_saves_distinct_files_and_resumes_by_seed(setup, monkeypatch):
    dls, store, _ = setup
    monkeypatch.setattr(ocm_module, "build_unet", lambda *a, **k: TinyModelStandIn())
    protocol = TrainProtocol(
        run_id="ocm-rgn-cs12-shared",
        architecture="ocm",
        learning_rate=1e-3,
        effective_batch_size=2,
    )

    first = ocm_module.run_ocm_ensemble_resumable(
        dls, protocol, seeds=(0,), store=store, progress=lambda _: None
    )
    second = ocm_module.run_ocm_ensemble_resumable(
        dls, protocol, seeds=(0, 1), store=store, progress=lambda _: None
    )

    assert len(first) == 2 and set(first.values()) == {"trained"}
    # seed 0 (both backbones) skipped, seed 1 (both backbones) trained
    assert sorted(second.values()) == ["skipped", "skipped", "trained", "trained"]
    assert len(second) == 4
    done = [name for name in second if store.is_done(name)]
    assert len(done) == 4                        # four different checkpoints on disk


def test_a_checkpoint_loads_back_into_an_equivalent_model(setup):
    dls, store, protocol = setup
    lr_search, winning = phase2.search_phase2_learning_rate(
        dls, protocol, TinyModelStandIn, search_seed=0
    )
    run = phase2.train_phase2_run(dls, winning, lr_search, 0, TinyModelStandIn)
    store.save_run(run)

    path = store.checkpoint_path(run.fit_result.checkpoint_name)
    restored = load_model_from_checkpoint(TinyModelStandIn, path)

    x = torch.rand(2, 3, 8, 8)
    trained = run.fit_result.learner.model.cpu().eval()
    assert torch.allclose(trained(x), restored(x), atol=1e-6)