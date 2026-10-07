"""Stop a run after any epoch and continue it later, in another session.

A full run is ten or more hours, longer than a Colab session may last. After every
epoch the whole state needed to go on is written to durable storage, and a run that
finds such a state continues from it instead of starting again.

fastai's own start_epoch cannot be used for this: it skips epochs by cancelling
them, and the callbacks that follow the validation loss then fail on the empty
record of an epoch that never ran. Here the epochs that remain are simply run, with
the learning-rate curve moved to the point where it was left.

What is carried over: the weights, the optimiser, the best weights of the phase so
far, the validation history and every random number generator. What is not: the
loss scale of mixed precision starts again (it adapts within a few steps), and the
gradients accumulated towards a step that was not finished are dropped.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from fastai.callback.core import Callback
from fastai.callback.schedule import ParamScheduler, combined_cos
from fastcore.foundation import L

STATE_VERSION = 1


@dataclass(frozen=True)
class PhaseSpec:
    """One of the two one-cycle fits that fastai's fine_tune runs in turn."""

    name: str
    epochs: int
    lr_max: object          # a float or a slice, as passed to fit_one_cycle
    pct_start: float
    div: float


def phase_specs(protocol) -> tuple[PhaseSpec, PhaseSpec]:
    """The two phases of Learner.fine_tune, exactly as it builds them.

    The names are fastai's. On a Learner without a splitter, which is how OCM's own
    notebook and this project build it, there is a single group of parameters:
    freeze() freezes nothing and the learning rate is one value for the whole
    network, base_lr in the first phase and base_lr / 2 in the second.
    """
    base_lr = protocol.learning_rate
    first = PhaseSpec(
        name="frozen",
        epochs=protocol.frozen_epochs,
        lr_max=slice(base_lr),
        pct_start=0.99,
        div=25.0,
    )
    half = base_lr / 2
    second = PhaseSpec(
        name="unfrozen",
        epochs=protocol.unfrozen_epochs,
        lr_max=slice(half / 100, half),
        pct_start=0.3,
        div=5.0,
    )
    return first, second


def continue_schedule(schedule, done: int, total: int):
    """The part of a schedule built for `total` epochs that is left after `done`.

    ParamScheduler asks its schedule for the position in the current fit, from 0 to
    1. A fit of the `total - done` remaining epochs must therefore ask the original
    schedule for positions from done / total up to 1.
    """
    start = done / total
    span = (total - done) / total
    return lambda position: schedule(start + position * span)


def fit_phase(learner, spec: PhaseSpec, done: int, weight_decay, callbacks=()) -> None:
    """Run the epochs of a phase from `done` on, as fit_one_cycle would have.

    With done = 0 this is fit_one_cycle itself, step by step.
    """
    if done >= spec.epochs:
        return
    if learner.opt is None:
        learner.create_opt()
    learner.opt.set_hyper("lr", spec.lr_max)
    lr_max = np.array([h["lr"] for h in learner.opt.hypers])
    schedules = {
        "lr": continue_schedule(
            combined_cos(spec.pct_start, lr_max / spec.div, lr_max, lr_max / 1e5),
            done,
            spec.epochs,
        ),
        "mom": continue_schedule(
            combined_cos(spec.pct_start, *learner.moms), done, spec.epochs
        ),
    }
    learner.fit(
        spec.epochs - done,
        cbs=ParamScheduler(schedules) + L(callbacks),
        reset_opt=False,
        wd=weight_decay,
    )


def capture_rng(resolution_cb) -> dict:
    """Every generator that decides what the next epoch will see."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "resolution": resolution_cb.rng.bit_generator.state,
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict, resolution_cb) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    resolution_cb.rng.bit_generator.state = state["resolution"]
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _copy_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """A CPU copy that never shares memory with the original. On the CPU a plain
    .cpu() returns the same storage, and a snapshot would then change as the
    model trains."""
    return tensor.detach().to("cpu", copy=True)


def cpu_state_dict(model: torch.nn.Module) -> dict:
    return {key: _copy_to_cpu(value) for key, value in model.state_dict().items()}


def _to_cpu(value):
    if isinstance(value, torch.Tensor):
        return _copy_to_cpu(value)
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_cpu(item) for item in value)
    return value


def optimizer_state(opt) -> dict:
    """fastai's optimiser state with every tensor on the CPU, ready to be saved."""
    return _to_cpu(opt.state_dict())


def load_optimizer_state(opt, state: dict) -> None:
    """Load a saved optimiser state and put its tensors where the parameters are.

    The model must already be on the device it will train on: the state follows the
    parameters, wherever they are when this runs.
    """
    opt.load_state_dict(state)
    for parameter, entry in opt.state.items():
        for key, value in entry.items():
            if isinstance(value, torch.Tensor):
                entry[key] = value.to(parameter.device)


class RestoreBest(Callback):
    """Give a resumed phase the best result it had reached before it was stopped.

    SaveModelCallback forgets its best value when a fit starts and the best weights
    file belongs to the session that is gone, so both are put back here, after the
    tracker has reset itself and before the first epoch runs.
    """

    order = 65  # after TrackerCallback (60) and SaveModelCallback (61)

    def __init__(self, save_cb, best: float, weights: dict | None, path: Path):
        super().__init__()
        self.save_cb = save_cb
        self.best = best
        self.weights = weights
        self.path = Path(path)

    def before_fit(self) -> None:
        self.save_cb.best = self.best
        if self.weights is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.weights, self.path)


class EpochSaver(Callback):
    """After every epoch, hand the full state to `save`."""

    order = 80  # after the recorder, the trackers and the history

    def __init__(self, snapshot, save, phase: str, already_done: int):
        super().__init__()
        self.snapshot = snapshot
        self.save = save
        self.phase = phase
        self.already_done = already_done

    def after_epoch(self) -> None:
        self.save(self.snapshot(self.phase, self.already_done + self.epoch + 1))