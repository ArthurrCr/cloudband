from dataclasses import replace

import numpy as np
import pytest
import torch
from fastai.callback.core import Callback
from fastai.learner import Learner
from fastai.torch_core import default_device

from cloudband.datasets.cloudsen12 import Sample
from cloudband.train.data import build_dataloaders
from cloudband.train.loop import fit_protocol
from cloudband.train.loss import build_loss
from cloudband.train.progress import continue_schedule, phase_specs
from cloudband.train.protocol import ocm_shared_protocol

FROZEN, UNFROZEN = 2, 3
TOTAL = FROZEN + UNFROZEN


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Conv2d(3, 8, kernel_size=1)
        self.second = torch.nn.Conv2d(8, 4, kernel_size=1)

    def forward(self, x):
        return self.second(torch.relu(self.first(x.float())))


def fake_read_sample(table, index, crop_to_valid=True):
    rng = np.random.default_rng(index)
    return Sample(
        identifier=f"scene-{index}",
        image=rng.integers(0, 10000, (13, 32, 32)).astype(np.int32),
        annotation=rng.integers(0, 4, (32, 32)).astype(np.int32),
    )


def make_protocol():
    # tiny resampled sizes (12 to 16 px), and 16 samples = 2 whole accumulation steps
    return replace(
        ocm_shared_protocol(1e-3),
        frozen_epochs=FROZEN,
        unfrozen_epochs=UNFROZEN,
        effective_batch_size=8,
        min_gsd_m=300.0,
        max_gsd_m=400.0,
        mixed_precision=False,
    )


def make_dls():
    return build_dataloaders(
        train_table=range(16),
        valid_table=range(8),
        micro_batch_size=4,
        read_sample=fake_read_sample,
    )


class MemoryStore:
    """Stands in for RunStore: keeps every saved state in memory."""

    def __init__(self, resume_from=None):
        self.resume_from = resume_from
        self.states = []
        self.heartbeats = 0

    def load_epoch_state(self, name):
        return self.resume_from

    def save_epoch_state(self, name, state):
        self.states.append(state)

    def heartbeat(self, name):
        self.heartbeats += 1


def run(tmp_path, monkeypatch, store, model_seed, protocol=None, extra_cbs=()):
    monkeypatch.chdir(tmp_path)
    torch.manual_seed(model_seed)
    learner = Learner(make_dls(), Tiny(), loss_func=build_loss())
    result = fit_protocol(
        learner, protocol or make_protocol(), seed=0, extra_cbs=extra_cbs,
        progress_store=store,
    )
    return learner, result


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    """One uninterrupted run that records its state after every epoch."""
    path = tmp_path_factory.mktemp("reference")
    mp = pytest.MonkeyPatch()
    store = MemoryStore()
    try:
        learner, result = run(path, mp, store, model_seed=1)
    finally:
        mp.undo()
    return learner, result, store


def same_weights(a, b):
    pairs = zip(a.state_dict().values(), b.state_dict().values())
    return all(torch.equal(x, y) for x, y in pairs)


def test_a_state_is_saved_after_every_epoch_and_at_the_phase_boundary(reference):
    _, _, store = reference

    assert len(store.states) == TOTAL + 1
    assert store.heartbeats == len(store.states)
    positions = [(s["phase"], s["epochs_done"]) for s in store.states]
    assert positions == [
        ("frozen", 1), ("frozen", 2), ("unfrozen", 0),
        ("unfrozen", 1), ("unfrozen", 2), ("unfrozen", 3),
    ]


@pytest.mark.parametrize("stopped_after", range(6))
def test_stopping_after_any_save_and_continuing_gives_the_same_model(
    reference, tmp_path, monkeypatch, stopped_after
):
    ref_learner, ref_result, ref_store = reference
    state = ref_store.states[stopped_after]

    # a new session: a new process would start from an untrained, differently
    # initialised model, so use another seed to prove the state restores everything
    learner, result = run(tmp_path, monkeypatch, MemoryStore(resume_from=state), 99)

    assert same_weights(learner.model, ref_learner.model)
    assert result.valid_losses == ref_result.valid_losses
    assert result.best_val_loss == ref_result.best_val_loss
    assert result.sampled_gsds == ref_result.sampled_gsds


def test_only_the_epochs_that_remain_are_run(reference, tmp_path, monkeypatch):
    _, _, ref_store = reference

    class CountEpochs(Callback):
        order = 90

        def __init__(self):
            self.n = 0

        def before_epoch(self):
            self.n += 1

    counter = CountEpochs()
    state = ref_store.states[3]            # ("unfrozen", 1): four epochs are done
    run(tmp_path, monkeypatch, MemoryStore(resume_from=state), 7, extra_cbs=(counter,))

    assert counter.n == TOTAL - (FROZEN + 1)


def test_progress_made_with_other_settings_is_refused(reference, tmp_path, monkeypatch):
    _, _, ref_store = reference
    other = replace(make_protocol(), learning_rate=1e-4)

    with pytest.raises(RuntimeError, match="other settings .*learning_rate"):
        run(tmp_path, monkeypatch, MemoryStore(resume_from=ref_store.states[1]), 1,
            protocol=other)


def test_progress_in_another_format_is_refused(reference, tmp_path, monkeypatch):
    _, _, ref_store = reference
    state = dict(ref_store.states[1], version=0)

    with pytest.raises(RuntimeError, match="format version"):
        run(tmp_path, monkeypatch, MemoryStore(resume_from=state), 1)


def test_a_saved_state_keeps_the_weights_of_its_own_epoch(reference):
    ref_learner, _, store = reference
    first = store.states[0]["model"]
    final = ref_learner.model.state_dict()

    # a snapshot that shared memory with the model would show the final weights here
    assert any(not torch.equal(first[key], final[key]) for key in first)


def test_the_schedule_is_continued_where_it_was_left():
    schedule = lambda position: position * 10

    rest = continue_schedule(schedule, done=3, total=10)

    assert rest(0.0) == pytest.approx(3.0)       # the position of epoch 4 of 10
    assert rest(1.0) == pytest.approx(10.0)      # and the end of the cycle
    assert rest(0.5) == pytest.approx(6.5)


def test_the_phases_are_the_ones_fine_tune_builds():
    first, second = phase_specs(make_protocol())

    assert (first.epochs, first.pct_start, first.div) == (FROZEN, 0.99, 25.0)
    assert (second.epochs, second.pct_start, second.div) == (UNFROZEN, 0.3, 5.0)
    assert first.lr_max == slice(1e-3)
    assert second.lr_max == slice(1e-3 / 2 / 100, 1e-3 / 2)

def test_the_model_is_on_its_device_before_the_optimiser_state_is_loaded(
    reference, tmp_path, monkeypatch
):
    # fastai moves the model to the GPU only when a fit starts; loading the state
    # before that left the momentum buffers on the CPU (a crash on the GPU that a
    # CPU-only run cannot show), so what is checked here is the order of the calls
    _, _, ref_store = reference
    state = next(s for s in ref_store.states if s["opt"] is not None)
    calls = []

    from cloudband.train import loop

    real_load = loop.load_optimizer_state

    def spy_load(opt, saved):
        calls.append("load_optimizer_state")
        return real_load(opt, saved)

    monkeypatch.setattr(loop, "load_optimizer_state", spy_load)
    monkeypatch.chdir(tmp_path)
    learner = Learner(make_dls(), Tiny(), loss_func=build_loss())
    real_to = learner.model.to

    def spy_to(*args, **kwargs):
        calls.append(("to", str(args[0])))
        return real_to(*args, **kwargs)

    monkeypatch.setattr(learner.model, "to", spy_to)

    fit_protocol(
        learner, make_protocol(), seed=0, progress_store=MemoryStore(resume_from=state)
    )

    device = str(getattr(learner.dls, "device", default_device()))
    assert ("to", device) in calls
    assert calls.index(("to", device)) < calls.index("load_optimizer_state")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_resuming_on_the_gpu_keeps_the_optimiser_state_on_the_gpu(
    reference, tmp_path, monkeypatch
):
    _, _, ref_store = reference
    state = next(s for s in ref_store.states if s["opt"] is not None)
    monkeypatch.chdir(tmp_path)
    dls = make_dls()
    dls.to(torch.device("cuda"))
    learner = Learner(dls, Tiny(), loss_func=build_loss())

    fit_protocol(
        learner, make_protocol(), seed=0, progress_store=MemoryStore(resume_from=state)
    )

    assert next(learner.model.parameters()).device.type == "cuda"
    for parameter, entry in learner.opt.state.items():
        for value in entry.values():
            if isinstance(value, torch.Tensor):
                assert value.device == parameter.device
