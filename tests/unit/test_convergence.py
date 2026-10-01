import json

import torch
from fastai.data.core import DataLoaders
from fastai.losses import CrossEntropyLossFlat
from fastai.learner import Learner

from cloudband.train.loop import fit_protocol
from cloudband.train.manifest import build_training_manifest, protocol_config
from cloudband.train.protocol import TrainProtocol
from cloudband.train.store import RunStore


class OneByOneConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def toy_learner():
    images = torch.rand(8, 3, 32, 32)
    annotations = torch.randint(0, 4, (8, 32, 32))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(images, annotations), batch_size=4
    )
    dls = DataLoaders(loader, loader)
    return Learner(dls, OneByOneConv(), loss_func=CrossEntropyLossFlat(axis=1))


def protocol(frozen, unfrozen):
    return TrainProtocol(
        run_id="ocm-rgn-cs12-shared",
        architecture="ocm",
        learning_rate=1e-3,
        frozen_epochs=frozen,
        unfrozen_epochs=unfrozen,
        effective_batch_size=4,
    )


def test_the_history_covers_both_the_frozen_and_unfrozen_phases(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = fit_protocol(toy_learner(), protocol(2, 3), seed=0)

    # fine_tune is two fits and the Recorder restarts at each, so reading it at the
    # end would give 3 values; the callback must give all 5
    assert len(result.valid_losses) == 5
    assert all(value > 0 for value in result.valid_losses)
    assert result.best_val_loss == min(result.valid_losses) or abs(
        result.best_val_loss - min(result.valid_losses)
    ) < 1e-6


def test_the_manifest_records_the_curve_and_the_best_epoch():
    config = protocol_config(protocol(1, 1), valid_losses=(0.9, 0.5, 0.7))

    assert config["valid_loss_history"] == [0.9, 0.5, 0.7]
    assert config["best_epoch"] == 1


def test_the_manifest_has_no_best_epoch_without_a_history():
    assert protocol_config(protocol(1, 1))["best_epoch"] is None


def test_a_real_fit_result_flows_into_the_manifest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = fit_protocol(toy_learner(), protocol(1, 1), seed=0)

    manifest = build_training_manifest(result, protocol(1, 1))

    assert len(manifest.config["valid_loss_history"]) == 2
    assert manifest.config["best_epoch"] in (0, 1)


def write_manifest(store, name, history):
    config = protocol_config(protocol(1, 1), valid_losses=history)
    store.manifest_path(name).write_text(json.dumps({"config": config}))


def test_convergence_flags_a_run_still_improving_at_the_end(tmp_path):
    store = RunStore(tmp_path / "r", tmp_path / "r" / "c", tmp_path / "m")
    write_manifest(store, "falling", [0.9, 0.8, 0.7, 0.6, 0.5, 0.4])

    summary = store.convergence("falling")

    assert summary["best_epoch"] == 5
    assert summary["still_improving"] is True


def test_convergence_accepts_a_run_that_peaked_early(tmp_path):
    store = RunStore(tmp_path / "r", tmp_path / "r" / "c", tmp_path / "m")
    write_manifest(store, "settled", [0.9, 0.5, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45])

    summary = store.convergence("settled")

    assert summary["best_epoch"] == 2
    assert summary["epochs"] == 8
    assert summary["still_improving"] is False


def test_convergence_is_none_without_a_manifest_or_history(tmp_path):
    store = RunStore(tmp_path / "r", tmp_path / "r" / "c", tmp_path / "m")
    assert store.convergence("missing") is None
    store.manifest_path("old").write_text(json.dumps({"config": {"run_id": "x"}}))
    assert store.convergence("old") is None