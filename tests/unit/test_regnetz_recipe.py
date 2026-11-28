import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from cloudband.datasets.cloudsen12 import Sample
from cloudband.train import regnetz_recipe as recipe_module
from cloudband.train.regnetz_recipe import (
    EarlyStopping,
    FocalTverskyLoss,
    RegNetZRecipe,
    build_regnetz_unet,
    loaders,
    to_canvas,
    train_all,
    train_seed,
)
from cloudband.train.store import RunStore

SMALL = replace(RegNetZRecipe(), canvas=32, max_epochs=4, batch_size=2, mixed_precision=False,
                seeds=(42, 43))


def test_the_recipe_is_the_earlier_repository_one():
    recipe = RegNetZRecipe()
    assert recipe.encoder == "tu-regnetz_d8" and recipe.encoder_weights == "imagenet"
    assert (recipe.in_channels, recipe.classes, recipe.canvas) == (13, 4, 512)
    assert recipe.focal_tversky == (0.3, 0.7, 0.75)
    assert (recipe.learning_rate, recipe.weight_decay, recipe.batch_size) == (1e-4, 1e-4, 4)
    assert (recipe.max_epochs, recipe.early_stopping_patience, recipe.min_delta) == (50, 5, 1e-4)
    assert (recipe.plateau_factor, recipe.plateau_patience, recipe.scale) == (0.1, 3, 10_000.0)
    assert recipe.seeds[0] == 42 and len(recipe.seeds) == 5
    assert recipe.checkpoint_name(42) == "unet-regnetz-d8-l1c13-ft-v1-seed42"


def test_the_network_takes_13_bands_on_the_512_canvas():
    model = build_regnetz_unet(RegNetZRecipe(), pretrained=False).eval()
    with torch.no_grad():
        assert model(torch.randn(1, 13, 64, 64)).shape == (1, 4, 64, 64)
    assert 27e6 < sum(p.numel() for p in model.parameters()) < 28.5e6


def test_the_509_window_goes_back_on_the_canvas_with_zeros_right_and_bottom():
    image = np.full((13, 509, 509), 7, np.uint16)
    annotation = np.full((509, 509), 2, np.uint8)
    canvas, label = to_canvas(image, annotation, RegNetZRecipe())
    assert canvas.shape == (13, 512, 512) and label.shape == (512, 512)
    assert (canvas[:, :509, :509] == 7).all() and not canvas[:, 509:, :].any()
    assert not canvas[:, :, 509:].any() and not label[509:, :].any() and not label[:, 509:].any()


def test_focal_tversky_matches_the_earlier_formula():
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 8, 8)
    targets = torch.randint(0, 4, (2, 8, 8))
    probs = logits.softmax(1)
    onehot = nn.functional.one_hot(targets, 4).permute(0, 3, 1, 2).float()
    tp = (probs * onehot).sum((0, 2, 3))
    fp = (probs * (1 - onehot)).sum((0, 2, 3))
    fn = ((1 - probs) * onehot).sum((0, 2, 3))
    tversky = (tp + 1e-5) / (tp + 0.3 * fp + 0.7 * fn + 1e-5)
    expected = ((1 - tversky) ** 0.75).mean()
    assert FocalTverskyLoss()(logits, targets).item() == pytest.approx(expected.item())
    perfect = nn.functional.one_hot(targets, 4).permute(0, 3, 1, 2).float() * 50
    assert FocalTverskyLoss()(perfect, targets).item() < 1e-3


def test_early_stopping_counts_epochs_without_a_real_improvement():
    stopper = EarlyStopping(patience=2, min_delta=0.01)
    assert not stopper.step(1.0)
    assert not stopper.step(0.995)      # less than min_delta: counts
    assert not stopper.step(0.9)        # improvement: resets
    assert not stopper.step(0.899)
    assert stopper.step(0.9)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(13, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 4, 1))

    def forward(self, x):
        return self.net(x)


def reader(table, index, crop=True):
    rng = np.random.default_rng(index)
    annotation = rng.integers(0, 4, (29, 29)).astype(np.uint8)
    image = (rng.integers(0, 3000, (13, 29, 29)) + annotation.astype(np.int64) * 2000).astype(np.uint16)
    return Sample(f"s{index}", image, annotation, "roi")


TABLE = pd.DataFrame({"tortilla:id": [f"id{i}" for i in range(6)]})


def data(seed):
    return loaders(TABLE, TABLE, reader, reader, SMALL, seed, num_workers=0)


def store_at(tmp_path):
    return RunStore(
        results_dir=tmp_path / "results", checkpoint_dir=tmp_path / "checkpoints",
        source_dir=tmp_path / "models",
    )


def test_a_seed_trains_saves_and_records_its_history(tmp_path):
    store = store_at(tmp_path)
    manifest = train_seed(SMALL, 42, data, store, "cpu", build=Tiny, log=lambda m: None)
    name = SMALL.checkpoint_name(42)
    assert store.is_done(name)
    assert store.load_epoch_state(name) is None
    saved = json.loads(store.manifest_path(name).read_text())
    assert saved["best_epoch"] == manifest["best_epoch"] >= 1
    assert len(saved["history"]["val_loss"]) == saved["epochs_run"] <= SMALL.max_epochs
    weights = torch.load(store.checkpoint_path(name), weights_only=True)
    Tiny().load_state_dict(weights)
    losses = saved["history"]["val_loss"]
    assert saved["best_val_loss"] == min(losses)


def test_an_interrupted_seed_resumes_and_ends_where_an_uninterrupted_one_does(
    tmp_path, monkeypatch
):
    straight = store_at(tmp_path / "straight")
    train_seed(SMALL, 42, data, straight, "cpu", build=Tiny, log=lambda m: None)
    expected = torch.load(straight.checkpoint_path(SMALL.checkpoint_name(42)), weights_only=True)
    expected_history = json.loads(straight.manifest_path(SMALL.checkpoint_name(42)).read_text())

    store = store_at(tmp_path / "broken")
    original = RunStore.save_epoch_state
    calls = []

    def save_then_fail(self, name, state):
        original(self, name, state)
        calls.append(state["epoch"])
        if state["epoch"] == 2:
            raise KeyboardInterrupt("session ended")

    monkeypatch.setattr(RunStore, "save_epoch_state", save_then_fail)
    with pytest.raises(KeyboardInterrupt):
        train_seed(SMALL, 42, data, store, "cpu", build=Tiny, log=lambda m: None)
    monkeypatch.setattr(RunStore, "save_epoch_state", original)

    resumed = []
    train_seed(SMALL, 42, data, store, "cpu", build=Tiny, log=resumed.append)
    assert any("resuming after epoch 2" in line for line in resumed)
    got = torch.load(store.checkpoint_path(SMALL.checkpoint_name(42)), weights_only=True)
    for key in expected:
        assert torch.equal(got[key], expected[key])
    history = json.loads(store.manifest_path(SMALL.checkpoint_name(42)).read_text())["history"]
    assert history["val_loss"] == pytest.approx(expected_history["history"]["val_loss"])


def test_every_seed_is_trained_once(tmp_path):
    store = store_at(tmp_path)
    trained = train_all(SMALL, data, store, "cpu", build=Tiny, log=lambda m: None)
    assert trained == [42, 43]
    again = train_all(SMALL, data, store, "cpu", build=lambda: pytest.fail("retrained"),
                      log=lambda m: None)
    assert again == []


def test_the_module_states_where_the_recipe_comes_from():
    assert "Cloud-Segmentation" in recipe_module.__doc__
    assert "no data augmentation" in recipe_module.__doc__


def test_the_molab_notebook_trains_the_recipe_and_sends_it_to_the_drive():
    import ast
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "notebooks/06_regnetz/train_regnetz_ft_molab.py"
    text = path.read_text()
    tree = ast.parse(text)
    assert any(
        isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "app"
        for node in tree.body
    )
    assert '"segmentation-models-pytorch"' in text
    assert "google.colab" not in text and "mo.md(" not in text
    assert 'experiment("l1c13")' in text
    assert "recipe = RegNetZRecipe()" in text
    assert "mo.stop(not train_button.value" in text
    assert "session.device" not in text
    assert text.index("drive_mirror.push_all()") > text.index("train_all(recipe, data, store, DEVICE)")
