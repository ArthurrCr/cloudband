"""U-Net + RegNetZ-D8 on 13 bands, trained with the recipe of the earlier repository.

The network proposed in github.com/ArthurrCr/Cloud-Segmentation ("UNet+RegNetZ-D8
(FT)") is trained here again, five seeds, to see whether its results come back. The
recipe is that repository's, as its code stood when the model was trained (December
2025), not the ADR-0023 protocol of the OCM and Swin comparison:

- segmentation_models_pytorch U-Net, encoder "tu-regnetz_d8" with ImageNet weights,
  13 input bands (smp repeats the RGB filters over the bands and rescales them), 4
  classes (models/U_NET.py there);
- Focal Tversky loss, alpha 0.3, beta 0.7, gamma 0.75 (losses/segmentation.py);
- Adam, learning rate 1e-4, weight decay 1e-4; ReduceLROnPlateau on the validation
  loss, factor 0.1, patience 3; at most 50 epochs, early stopping after 5 epochs
  without an improvement of 1e-4; the checkpoint is the epoch of lowest validation
  loss; batch 4; mixed precision (training/trainer.py, experiment/config.py);
- the image divided by 10,000, no other normalisation, and no data augmentation;
- the whole 512 x 512 canvas, its 3 rows and columns of padding included (zeros in
  the image, clear sky in the label), L1C only, high-quality labels, p509 patches.

The learning rate of the final Focal Tversky model is not recorded in that repository;
1e-4 is its default and the rate of its loss comparison (stage 1). The cache holds the
509 window cut by the fixed crop (DIVERGENCES D17); padding it back to 512 with zeros
on the right and bottom gives exactly the canvas the earlier code read.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

RECIPE_REVISION = "v1"


@dataclass(frozen=True)
class RegNetZRecipe:
    run_id: str = f"unet-regnetz-d8-l1c13-ft-{RECIPE_REVISION}"
    encoder: str = "tu-regnetz_d8"
    encoder_weights: str | None = "imagenet"
    in_channels: int = 13
    classes: int = 4
    focal_tversky: tuple[float, float, float] = (0.3, 0.7, 0.75)  # alpha, beta, gamma
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 4
    max_epochs: int = 50
    early_stopping_patience: int = 5
    min_delta: float = 1e-4
    plateau_factor: float = 0.1
    plateau_patience: int = 3
    scale: float = 10_000.0
    canvas: int = 512
    mixed_precision: bool = True
    # 42 is the seed of the earlier run; the other four follow it.
    seeds: tuple[int, ...] = field(default=(42, 43, 44, 45, 46))

    def checkpoint_name(self, seed: int) -> str:
        return f"{self.run_id}-seed{seed}"


# ---- model and loss -------------------------------------------------------------


def build_regnetz_unet(recipe: RegNetZRecipe, pretrained: bool = True) -> nn.Module:
    import segmentation_models_pytorch as smp

    return smp.Unet(
        encoder_name=recipe.encoder,
        encoder_weights=recipe.encoder_weights if pretrained else None,
        in_channels=recipe.in_channels,
        classes=recipe.classes,
    )


class FocalTverskyLoss(nn.Module):
    """Focal Tversky loss, as in the earlier repository (losses/segmentation.py)."""

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 0.75,
                 smooth: float = 1e-5) -> None:
        super().__init__()
        self.alpha, self.beta, self.gamma, self.smooth = alpha, beta, gamma, smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        if targets.dim() == 4:
            targets = targets.squeeze(1)
        probs = F.softmax(logits, dim=1)
        onehot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        tp = (probs * onehot).sum(dims)
        fp = (probs * (1 - onehot)).sum(dims)
        fn = ((1 - probs) * onehot).sum(dims)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return ((1.0 - tversky) ** self.gamma).mean()


# ---- data -----------------------------------------------------------------------


def to_canvas(image: np.ndarray, annotation: np.ndarray, recipe: RegNetZRecipe):
    """The 509 window back on the 512 canvas: zeros right and bottom, as the files hold."""
    rows, columns = annotation.shape[-2:]
    pad_rows, pad_columns = recipe.canvas - rows, recipe.canvas - columns
    if pad_rows < 0 or pad_columns < 0:
        raise ValueError(f"a {rows}x{columns} patch does not fit a {recipe.canvas} canvas")
    image = np.pad(image, ((0, 0), (0, pad_rows), (0, pad_columns)))
    annotation = np.pad(annotation, ((0, pad_rows), (0, pad_columns)))
    return image, annotation


def model_input(image: np.ndarray, recipe: RegNetZRecipe) -> torch.Tensor:
    """What the network reads: the bands as float, divided by 10,000."""
    return torch.from_numpy(image.astype(np.float32) / recipe.scale)


class CanvasDataset(Dataset):
    """Samples of a cache as the earlier code read them: whole canvas, scaled, as is."""

    def __init__(self, table, read_sample, recipe: RegNetZRecipe):
        self.table, self.read_sample, self.recipe = table, read_sample, recipe

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int):
        sample = self.read_sample(self.table, index, True)
        image, annotation = to_canvas(sample.image, sample.annotation, self.recipe)
        return model_input(image, self.recipe), torch.from_numpy(annotation.astype(np.int64))


def loaders(train_table, valid_table, train_reader, valid_reader, recipe: RegNetZRecipe,
            seed: int, num_workers: int = 2):
    generator = torch.Generator().manual_seed(seed)
    train = DataLoader(
        CanvasDataset(train_table, train_reader, recipe), batch_size=recipe.batch_size,
        shuffle=True, num_workers=num_workers, generator=generator,
        persistent_workers=num_workers > 0,
    )
    valid = DataLoader(
        CanvasDataset(valid_table, valid_reader, recipe), batch_size=recipe.batch_size,
        shuffle=False, num_workers=num_workers, persistent_workers=num_workers > 0,
    )
    return train, valid


def predict_classes(model: nn.Module, image: np.ndarray, recipe: RegNetZRecipe,
                    device: str = "cpu") -> np.ndarray:
    """Class map of one 509 window, read as in training: on the canvas, divided by 10,000.

    The window is padded back to the 512 canvas, the network runs on it, and the
    prediction is cut back to the window, so the padding never reaches a score.
    """
    rows, columns = image.shape[-2:]
    canvas, _ = to_canvas(image, np.zeros((rows, columns), np.uint8), recipe)
    tensor = model_input(canvas, recipe).unsqueeze(0).to(device)
    with torch.no_grad():
        classes = model(tensor).argmax(dim=1).squeeze(0).cpu().numpy()
    return classes[:rows, :columns]


def load_trained(recipe: RegNetZRecipe, path: Path, device: str = "cpu") -> nn.Module:
    """A trained seed: the architecture without downloading ImageNet weights, then its own."""
    model = build_regnetz_unet(recipe, pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.to(device).eval()


# ---- training -------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """As the earlier code did: every generator, and deterministic cuDNN."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class EarlyStopping:
    patience: int
    min_delta: float
    counter: int = 0
    best: float | None = None

    def step(self, value: float) -> bool:
        """True when training should stop (the earlier callbacks.EarlyStopping, mode min)."""
        if self.best is None:
            self.best = value
            return False
        if self.best - value < self.min_delta:
            self.counter += 1
        else:
            self.best, self.counter = value, 0
        return self.counter >= self.patience


@dataclass
class RecipeFit:
    """The finished run, shaped as RunStore.save_run expects it."""

    checkpoint_name: str
    manifest_data: dict

    @property
    def fit_result(self):
        return self

    @property
    def manifest(self):
        return self

    def write(self, path: Path) -> Path:
        partial = Path(path).with_suffix(".json.partial")
        partial.write_text(json.dumps(self.manifest_data, indent=2))
        partial.replace(path)
        return Path(path)


def _rng_state(generator: torch.Generator) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader": generator.get_state(),
    }


def _set_rng_state(state: dict, generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["loader"])


def train_seed(
    recipe: RegNetZRecipe,
    seed: int,
    data: Callable[[int], tuple[DataLoader, DataLoader]],
    store,
    device: str = "cuda",
    build: Callable[[], nn.Module] | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Train one seed, resuming from the state saved after its last finished epoch.

    The state (weights, optimizer, scaler, scheduler, early stopping, history and
    every random generator) is saved after each epoch with store.save_epoch_state,
    so a session that ends in the middle loses at most the epoch in progress. The
    best weights go to store.source_dir and, at the end, through store.save_run.
    """
    name = recipe.checkpoint_name(seed)
    set_seed(seed)
    model = (build or (lambda: build_regnetz_unet(recipe)))().to(device)
    train_loader, valid_loader = data(seed)
    generator = train_loader.generator
    criterion = FocalTverskyLoss(*recipe.focal_tversky)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=recipe.learning_rate, weight_decay=recipe.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=recipe.plateau_factor, patience=recipe.plateau_patience
    )
    use_amp = recipe.mixed_precision and str(device).startswith("cuda")
    scaler = torch.amp.GradScaler(enabled=use_amp)
    stopper = EarlyStopping(recipe.early_stopping_patience, recipe.min_delta)
    history = {"train_loss": [], "val_loss": [], "lr": [], "seconds": []}
    best_loss, best_epoch, start = float("inf"), 0, 0
    best_path = Path(store.source_dir) / f"{name}.pth"
    best_path.parent.mkdir(parents=True, exist_ok=True)

    saved = store.load_epoch_state(name)
    if saved is not None:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        scheduler.load_state_dict(saved["scheduler"])
        stopper = EarlyStopping(**saved["early_stopping"])
        history, best_loss, best_epoch = saved["history"], saved["best_loss"], saved["best_epoch"]
        torch.save(saved["best_model"], best_path)
        _set_rng_state(saved["rng"], generator)
        start = saved["epoch"]
        log(f"{name}: resuming after epoch {start}, best validation loss {best_loss:.4f} "
            f"at epoch {best_epoch}")
    best_state = saved["best_model"] if saved is not None else None

    stopped = saved is not None and saved.get("stopped", False)
    for epoch in range(start + 1, recipe.max_epochs + 1):
        if stopped:
            break
        started = time.perf_counter()
        model.train()
        total = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                loss = criterion(model(images), masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()
        train_loss = total / len(train_loader)

        model.eval()
        total = 0.0
        with torch.no_grad():
            for images, masks in valid_loader:
                images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    total += criterion(model(images), masks).item()
        val_loss = total / len(valid_loader)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["seconds"].append(time.perf_counter() - started)
        scheduler.step(val_loss)
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, best_path)
        stopped = stopper.step(val_loss)
        log(f"{name}: epoch {epoch}/{recipe.max_epochs}  train {train_loss:.4f}  "
            f"valid {val_loss:.4f}  best {best_loss:.4f} (epoch {best_epoch})  "
            f"{history['seconds'][-1] / 60:.1f} min" + ("  early stop" if stopped else ""))

        store.save_epoch_state(name, {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "scheduler": scheduler.state_dict(),
            "early_stopping": asdict(stopper),
            "history": history,
            "best_loss": best_loss,
            "best_epoch": best_epoch,
            "best_model": best_state,
            "rng": _rng_state(generator),
            "stopped": stopped,
        })
        heartbeat = getattr(store, "heartbeat", None)
        if heartbeat is not None:
            heartbeat(name)

    manifest = {
        "checkpoint_name": name,
        "recipe": {**asdict(recipe), "focal_tversky": list(recipe.focal_tversky)},
        "seed": seed,
        "epochs_run": len(history["val_loss"]),
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "stopped_early": bool(stopped),
        "history": history,
        "source": "github.com/ArthurrCr/Cloud-Segmentation, December 2025",
    }
    store.save_run(RecipeFit(name, manifest))
    store.clear_epoch_state(name)
    return manifest


def train_all(
    recipe: RegNetZRecipe,
    data: Callable[[int], tuple[DataLoader, DataLoader]],
    store,
    device: str = "cuda",
    build: Callable[[], nn.Module] | None = None,
    log: Callable[[str], None] = print,
) -> list[int]:
    """Every seed not done and not taken by another session, one after the other."""
    trained = []
    for seed in recipe.seeds:
        name = recipe.checkpoint_name(seed)
        if store.is_done(name):
            continue
        if not store.claim(name):
            log(f"{name}: taken by {store.describe_lock(name)}; skipped")
            continue
        try:
            train_seed(recipe, seed, data, store, device, build, log)
            trained.append(seed)
        finally:
            store.release(name)
    left = [s for s in recipe.seeds if not store.is_done(recipe.checkpoint_name(s))]
    log(f"trained now: {trained}; seeds still to train: {left}")
    return trained
