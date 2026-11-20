import subprocess
import sys
import types

import numpy as np
import pandas as pd
import pytest
import torch

from cloudband.datasets.cloudsen12 import Sample
from cloudband.models.ocm import PAPER_BACKBONES, build_unet
from cloudband.models.swin_upernet import (
    OUTPUT_CLASSES,
    adapt_patch_embedding,
    build_swin_upernet,
)
from cloudband.pipelines import phase2_eval
from cloudband.train import experiments
from cloudband.train.cache import (
    BYTES_PER_SAMPLE,
    LocalCache,
    bytes_per_sample,
    measured_sample_bytes,
)
from cloudband.train.protocol import (
    L1C_BANDS,
    PROTOCOL_REVISION,
    ocm_l1c13_protocol,
    ocm_shared_protocol,
    swin_l1c13_protocol,
    swin_shared_protocol,
)
from cloudband.train.smoke import _fits_in_memory

SIZE = 64


def test_the_l1c_product_has_thirteen_bands_in_file_order():
    assert L1C_BANDS == (
        "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12",
    )
    rgn = ocm_shared_protocol(1e-4)
    assert tuple(L1C_BANDS[i] for i in rgn.band_indices) == tuple(rgn.bands)


@pytest.mark.parametrize("make, shared", [
    (ocm_l1c13_protocol, ocm_shared_protocol),
    (swin_l1c13_protocol, swin_shared_protocol),
])
def test_the_13_band_protocols_change_only_the_input(make, shared):
    new, old = make(3e-4), shared(3e-4)
    assert new.run_id != old.run_id and new.run_id.endswith(PROTOCOL_REVISION)
    assert new.bands == L1C_BANDS and new.band_indices == tuple(range(13))
    assert new.processing_levels == ("L1C",)
    differ = {
        name for name in vars(old)
        if getattr(old, name) != getattr(new, name)
    }
    assert differ == {"run_id", "bands", "band_indices", "processing_levels"}


def test_every_ocm_backbone_takes_thirteen_bands():
    for backbone in PAPER_BACKBONES:
        model = build_unet(backbone, pretrained=False, in_chans=13).eval()
        with torch.no_grad():
            assert model(torch.randn(1, 13, SIZE, SIZE)).shape == (1, 4, SIZE, SIZE)


def test_swin_upernet_takes_thirteen_bands():
    model = build_swin_upernet(pretrained=False, in_chans=13).eval()
    with torch.no_grad():
        output = model(torch.randn(1, 13, SIZE, SIZE))
    assert output.shape == (1, OUTPUT_CLASSES, SIZE, SIZE)


def test_the_swin_patch_embedding_is_adapted_as_timm_adapts_the_cnns():
    from timm.models import adapt_input_conv

    backbone = build_swin_upernet(pretrained=False).model.backbone

    def projection_of(module):
        return next(m for m in module.modules() if type(m).__name__ == "SwinPatchEmbeddings").projection

    rgb = projection_of(backbone).weight.detach().clone()
    bias = projection_of(backbone).bias.detach().clone()

    adapt_patch_embedding(backbone, 13)

    projection = projection_of(backbone)
    assert projection.in_channels == 13
    assert torch.equal(projection.weight, adapt_input_conv(13, rgb))
    assert torch.equal(projection.bias, bias)
    assert backbone.config.num_channels == 13
    # timm's rule: the RGB filters repeated over the bands, scaled by 3 / 13
    assert torch.allclose(projection.weight[:, :3], rgb * 3 / 13)
    assert torch.allclose(projection.weight[:, 3:6], rgb * 3 / 13)
    with torch.no_grad():
        backbone(torch.randn(1, 13, SIZE, SIZE))


def test_an_adapted_swin_has_the_same_layers_as_one_built_for_thirteen_bands():
    adapted = build_swin_upernet(pretrained=False)
    adapt_patch_embedding(adapted.model.backbone, 13)
    native = build_swin_upernet(pretrained=False, in_chans=13)
    shapes = {k: v.shape for k, v in adapted.state_dict().items()}
    assert shapes == {k: v.shape for k, v in native.state_dict().items()}


def test_the_memory_probe_feeds_the_protocol_band_count():
    seen = []

    class Probe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(13, 4, 1)

        def forward(self, x):
            seen.append(x.shape[1])
            return self.conv(x)

    assert _fits_in_memory(Probe, 2, 8, False, 0.85, device="cpu", channels=13)
    assert seen == [13]


def test_the_cache_space_check_scales_with_the_bands():
    assert bytes_per_sample(3) <= BYTES_PER_SAMPLE
    assert 6.8e6 < bytes_per_sample(13) < 7.5e6


def test_the_measured_size_matches_the_files_build_writes(tmp_path):
    table = pd.DataFrame({"tortilla:id": [f"id{i}" for i in range(3)]})
    samples = [thirteen_band_sample(table, i) for i in range(3)]
    for compress in (False, True):
        cache = LocalCache(tmp_path / str(compress))
        cache.build("train", table, read_sample=thirteen_band_sample, bands_selected=True,
                    progress=lambda _: None, workers=1, compress=compress)
        written = max(p.stat().st_size for p in (tmp_path / str(compress) / "train").glob("*.npz"))
        measured = measured_sample_bytes(samples, bands_selected=True, compress=compress,
                                         headroom=1.0)
        assert measured == written
    smooth = Sample("s", np.zeros((13, 24, 24), np.uint16), np.zeros((24, 24), np.uint8), "r")
    assert measured_sample_bytes([smooth], True, compress=True) < measured_sample_bytes(
        [smooth], True, compress=False
    )
    with pytest.raises(ValueError):
        measured_sample_bytes([])


def thirteen_band_sample(table, index, crop_to_valid=True):
    rng = np.random.default_rng(index)
    return Sample(
        identifier=f"scene-{index}",
        image=rng.integers(0, 10000, (13, 24, 24)).astype(np.uint16),
        annotation=rng.integers(0, 4, (24, 24)).astype(np.uint8),
        roi_id="roi",
    )


@pytest.mark.parametrize("compress", [False, True])
def test_a_cache_of_every_band_reads_back_unchanged(tmp_path, compress):
    table = pd.DataFrame({"tortilla:id": [f"id{i}" for i in range(4)]})
    cache = LocalCache(tmp_path)
    cache.build("train", table, read_sample=thirteen_band_sample, bands_selected=True,
                progress=lambda _: None, workers=2, compress=compress,
                sample_bytes=bytes_per_sample(13))
    reader = cache.reader("train", table)
    for index in range(4):
        got, expected = reader(table, index), thirteen_band_sample(table, index)
        assert got.image.shape == (13, 24, 24)
        assert np.array_equal(got.image, expected.image)
        assert np.array_equal(got.annotation, expected.annotation)


def test_the_gdal_check_compares_every_band_when_asked(monkeypatch):
    table = pd.DataFrame({"a": range(3)})
    full = thirteen_band_sample(table, 0)
    monkeypatch.setattr(phase2_eval, "gdal_read_sample", lambda table, row: full)

    assert phase2_eval.check_fast_reader(lambda t, row: full, {"test": table}, None) == 2
    three = types.SimpleNamespace(image=full.image[[3, 2, 8]], annotation=full.annotation)
    with pytest.raises(AssertionError, match="differs from GDAL"):
        phase2_eval.check_fast_reader(lambda t, row: three, {"test": table}, None)


def test_the_rgn_experiment_keeps_the_phase2_names():
    rgn = experiments.experiment("rgn")
    ocm, swin = rgn.protocols()
    assert ocm.run_id == ocm_shared_protocol(1e-4).run_id
    assert swin.run_id == swin_shared_protocol(1e-4).run_id
    assert rgn.levels == ("L1C", "L2A")
    assert rgn.cache_dir_name == "cloudband_cache"
    assert rgn.micro_batch_setting == "micro_batch"
    assert rgn.band_count == 3


def test_the_l1c13_experiment_has_its_own_cache_and_micro_batch():
    chosen = experiments.experiment("l1c13")
    ocm, swin = chosen.protocols()
    assert ocm.run_id == ocm_l1c13_protocol(1e-4).run_id
    assert swin.run_id == swin_l1c13_protocol(1e-4).run_id
    assert chosen.levels == ("L1C",)
    assert chosen.band_count == 13
    assert chosen.reader_bands is None
    names = [(e.cache_dir_name, e.micro_batch_setting) for e in experiments.EXPERIMENTS.values()]
    assert len(set(names)) == len(names)


def test_an_unknown_experiment_is_refused():
    with pytest.raises(ValueError, match="unknown experiment"):
        experiments.experiment("rgb")


def test_the_tables_come_from_the_experiment_levels_l1c_first():
    loaded = []
    fake = types.SimpleNamespace(
        REMOTE_L1C="l1c",
        REMOTE_L2A="l2a",
        load_remote=lambda source: loaded.append(source) or pd.DataFrame(
            {"level": [source] * 4, "split": ["train", "train", "validation", "test"]}
        ),
        select=lambda table, split: table[table.split == split],
    )
    for name, levels in (("rgn", ["l1c", "l2a"]), ("l1c13", ["l1c"])):
        loaded.clear()
        chosen = experiments.experiment(name)
        tables = experiments.load_tables(chosen, fake)
        assert loaded == levels
        train = chosen.tables(tables["train"])
        assert list(train.level) == [level for level in levels for _ in range(2)]
        assert list(train.index) == list(range(len(train)))
        assert len(chosen.tables(tables["validation"])) == len(levels)


def test_workers_are_told_which_experiment_to_train(tmp_path, monkeypatch):
    from cloudband.molab import workers

    commands = []

    class FakeProcess:
        def __init__(self, command, **kwargs):
            commands.append(command)

    monkeypatch.setattr(workers.subprocess, "Popen", FakeProcess)
    workers.launch(2, tmp_path, tmp_path, tmp_path, tmp_path / "t.json", experiment="l1c13")
    for command in commands:
        assert command[command.index("--experiment") + 1] == "l1c13"
    monkeypatch.undo()

    reply = subprocess.run(
        [sys.executable, "-m", "cloudband.molab.workers", "--help"],
        capture_output=True, text=True, env={"PYTHONPATH": "src"}, check=False,
    )
    assert "--experiment" in reply.stdout
