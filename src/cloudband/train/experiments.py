"""The training experiments of the comparison, and what each one needs from a session.

An experiment fixes the input of both architectures: which bands, which processing
levels of CloudSEN12+, and so which local cache and which micro-batch (13 bands take
more GPU memory than 3). Everything else, the protocol, the learning-rate grid and the
seeds, is the shared one of ADR-0023, so the experiments differ only in their input.

- "rgn": red, green and near-infrared (B04, B03, B8A), trained on L1C and L2A. The
  phase 2 comparison, whose models are already trained.
- "l1c13": all 13 bands of the L1C product, trained on L1C only, since L2A has no B10
  (DIVERGENCES D18).

The molab notebook and its parallel workers read the same entry, so a worker always
trains what the notebook set up.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from cloudband.pipelines.phase0 import RGN_INDICES
from cloudband.train.cache import bytes_per_sample
from cloudband.train.protocol import (
    TrainProtocol,
    ocm_l1c13_protocol,
    ocm_shared_protocol,
    swin_l1c13_protocol,
    swin_shared_protocol,
)

# The learning rate the protocols are built with before the search picks one; the
# scheduler replaces it with the winner of each network's search.
PLACEHOLDER_LEARNING_RATE = 1e-4


@dataclass(frozen=True)
class Experiment:
    name: str
    description: str
    ocm: Callable[[float], TrainProtocol]
    swin: Callable[[float], TrainProtocol]
    reader_bands: tuple[int, ...] | None
    cache_dir_name: str
    micro_batch_setting: str

    def protocols(
        self, learning_rate: float = PLACEHOLDER_LEARNING_RATE
    ) -> tuple[TrainProtocol, TrainProtocol]:
        """The OCM and the Swin+UPerNet protocol, in that order."""
        return self.ocm(learning_rate), self.swin(learning_rate)

    @property
    def levels(self) -> tuple[str, ...]:
        return tuple(self.ocm(PLACEHOLDER_LEARNING_RATE).processing_levels)

    @property
    def band_count(self) -> int:
        return len(self.ocm(PLACEHOLDER_LEARNING_RATE).bands)

    @property
    def sample_bytes(self) -> int:
        """Disk one uncompressed cached sample takes."""
        return bytes_per_sample(self.band_count)

    def remote_sources(self, dataset) -> dict[str, str]:
        """The CloudSEN12+ archive of each processing level, in training order."""
        sources = {"L1C": dataset.REMOTE_L1C, "L2A": dataset.REMOTE_L2A}
        return {level: sources[level] for level in self.levels}

    def tables(
        self, by_level: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """The training (or validation) table: the levels one after the other, L1C first.

        The order is part of the cache's fingerprint, so it never changes.
        """
        return pd.concat([by_level[level] for level in self.levels], ignore_index=True)


EXPERIMENTS: dict[str, Experiment] = {
    "rgn": Experiment(
        name="rgn",
        description="R-G-NIR (B04, B03, B8A), L1C and L2A",
        ocm=ocm_shared_protocol,
        swin=swin_shared_protocol,
        reader_bands=RGN_INDICES,
        cache_dir_name="cloudband_cache",
        micro_batch_setting="micro_batch",
    ),
    "l1c13": Experiment(
        name="l1c13",
        description="all 13 L1C bands (B01 to B12, with B8A and B10), L1C only",
        ocm=ocm_l1c13_protocol,
        swin=swin_l1c13_protocol,
        reader_bands=None,
        cache_dir_name="cloudband_cache_l1c13",
        micro_batch_setting="micro_batch_l1c13",
    ),
}


def experiment(name: str) -> Experiment:
    try:
        return EXPERIMENTS[name]
    except KeyError:
        raise ValueError(f"unknown experiment {name!r}; known: {sorted(EXPERIMENTS)}") from None


def load_tables(
    chosen: Experiment, dataset, splits: tuple[str, ...] = ("train", "validation")
) -> dict[str, dict[str, pd.DataFrame]]:
    """Each split of each level the experiment uses, read from the remote archives."""
    tables: dict[str, dict[str, pd.DataFrame]] = {split: {} for split in splits}
    for level, source in chosen.remote_sources(dataset).items():
        full = dataset.load_remote(source)
        for split in splits:
            tables[split][level] = dataset.select(full, split=split)
    return tables
