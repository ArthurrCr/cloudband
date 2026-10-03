"""A local copy of the samples the models use, so epochs do not wait on the network.

Reading one CloudSEN12+ patch from the remote archive was measured at about six
seconds, twenty times what the GPU needs per sample, and it pulls all thirteen
bands when only three are used. Training straight from the archive would leave
the GPU idle for hours of every epoch. The cache keeps the red, green and
near-infrared bands, already cropped to the valid window, plus the annotation,
about 1.8 MB per patch instead of 6.7 MB, on the session's local disk.

A cache belongs to one exact table. Its fingerprint is stored beside the files
and checked on every read, so a table that was reordered, or whose L1C and L2A
halves were swapped, is refused instead of silently feeding the wrong patches.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from cloudband.datasets import cloudsen12 as dataset
from cloudband.datasets.cloudsen12 import Sample
from cloudband.pipelines.phase0 import select_rgn

META_FILE = "meta.json"
BYTES_PER_SAMPLE = 2_000_000  # 3 bands + annotation at 509 x 509, with headroom

SampleReader = Callable[[pd.DataFrame, int, bool], Sample]


def table_fingerprint(table: pd.DataFrame) -> str:
    """A hash of every cell of the table, in order, so any change shows up."""
    text = pd.DataFrame({column: table[column].astype(str) for column in table.columns})
    hashed = pd.util.hash_pandas_object(text, index=False)
    return hashlib.sha256(hashed.to_numpy().tobytes()).hexdigest()


def _sample_path(directory: Path, index: int) -> Path:
    return directory / f"{index:06d}.npz"


def _write_sample(path: Path, sample: Sample, bands_selected: bool = False) -> None:
    partial = path.with_suffix(".partial")
    image = sample.image if bands_selected else select_rgn(sample.image)
    with open(partial, "wb") as handle:
        np.savez(
            handle,
            image=np.ascontiguousarray(image),
            annotation=sample.annotation,
            identifier=np.array(sample.identifier),
            roi_id=np.array(sample.roi_id or ""),
        )
    partial.replace(path)


class CachedReader:
    """Reads samples back from a cache. A class, not a closure, so it pickles."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def __call__(
        self, table: pd.DataFrame, index: int, crop_to_valid: bool = True
    ) -> Sample:
        if not crop_to_valid:
            raise ValueError("the cache holds samples cropped to the valid window only")
        with np.load(_sample_path(self.directory, index)) as data:
            roi_id = str(data["roi_id"])
            return Sample(
                identifier=str(data["identifier"]),
                image=data["image"],
                annotation=data["annotation"],
                roi_id=roi_id or None,
            )


class SubsetReader:
    """Maps positions in a slice of a table back to positions in the cached table."""

    def __init__(self, reader: SampleReader, positions: list[int]):
        self.reader = reader
        self.positions = list(positions)

    def __call__(
        self, table: pd.DataFrame, index: int, crop_to_valid: bool = True
    ) -> Sample:
        return self.reader(table, self.positions[index], crop_to_valid)


def subset_reader(reader: SampleReader, positions: list[int]) -> SubsetReader:
    return SubsetReader(reader, positions)


class LocalCache:
    """Per-split caches under one directory, built once and read many times."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def split_dir(self, name: str) -> Path:
        return self.directory / name

    def _check_table(self, directory: Path, table: pd.DataFrame) -> None:
        meta_path = directory / META_FILE
        stored = json.loads(meta_path.read_text())
        if stored["fingerprint"] != table_fingerprint(table):
            raise ValueError(
                f"the cache in {directory} was built for a different table, in "
                "another order or with other rows; delete it and build again"
            )

    def build(
        self,
        name: str,
        table: pd.DataFrame,
        read_sample: SampleReader | None = None,
        workers: int = 16,
        retries: int = 4,
        backoff_seconds: float = 4.0,
        progress: Callable[[str], None] = print,
        bands_selected: bool = False,
    ) -> None:
        """Copy every sample of the table, in parallel, skipping those already copied.

        Safe to run again after an interruption: finished files are kept, and a
        file is only ever renamed into place once fully written. Raises when
        some samples still fail after the retries; running it again retries them.

        bands_selected says the reader already returns only the red, green and
        near-infrared bands, as datasets.fast_read.FastReader does by default.
        """
        reader = read_sample if read_sample is not None else dataset.read_sample
        directory = self.split_dir(name)
        directory.mkdir(parents=True, exist_ok=True)

        meta_path = directory / META_FILE
        fingerprint = table_fingerprint(table)
        if meta_path.is_file():
            self._check_table(directory, table)
        else:
            meta = {"fingerprint": fingerprint, "n": len(table)}
            meta_path.write_text(json.dumps(meta))

        pending = [
            i for i in range(len(table)) if not _sample_path(directory, i).is_file()
        ]
        if not pending:
            progress(f"{name}: already complete, {len(table)} samples")
            return

        needed = len(pending) * BYTES_PER_SAMPLE
        free = shutil.disk_usage(directory).free
        if free < needed:
            raise RuntimeError(
                f"{name}: needs about {needed / 1e9:.1f} GB, "
                f"only {free / 1e9:.1f} GB free"
            )

        def fetch(index: int) -> int:
            for attempt in range(retries + 1):
                try:
                    _write_sample(
                        _sample_path(directory, index),
                        reader(table, index, True),
                        bands_selected,
                    )
                    return index
                except Exception:
                    if attempt == retries:
                        raise
                    time.sleep(backoff_seconds * (attempt + 1))
            return index

        progress(
            f"{name}: copying {len(pending)} of {len(table)} samples, "
            f"{workers} threads"
        )
        started = time.perf_counter()
        failed: list[int] = []
        step = max(1, len(pending) // 40)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(fetch, index): index for index in pending}
            for done, future in enumerate(as_completed(futures), start=1):
                if future.exception() is not None:
                    failed.append(futures[future])
                if done % step == 0 or done == len(pending):
                    rate = done / (time.perf_counter() - started)
                    remaining = (len(pending) - done) / rate / 60
                    progress(
                        f"{name}: {done}/{len(pending)}  {rate:.1f} samples/s  "
                        f"about {remaining:.0f} min left"
                    )
        if failed:
            raise RuntimeError(
                f"{name}: {len(failed)} samples failed after {retries} retries "
                f"(first: {sorted(failed)[:5]}); run build again to retry them"
            )

    def reader(self, name: str, table: pd.DataFrame) -> CachedReader:
        """A reader for the table, after checking the cache is complete and matches."""
        directory = self.split_dir(name)
        if not (directory / META_FILE).is_file():
            raise FileNotFoundError(
                f"no cache for {name!r} in {directory}; build it first"
            )
        self._check_table(directory, table)
        present = sum(1 for _ in directory.glob("*.npz"))
        if present != len(table):
            raise RuntimeError(
                f"the cache for {name!r} has {present} of {len(table)} samples; "
                "run build again to finish it"
            )
        return CachedReader(directory)

    def samples(self, name: str, table: pd.DataFrame) -> Iterator[Sample]:
        """Every sample of the table in order, for scoring."""
        reader = self.reader(name, table)
        for index in range(len(table)):
            yield reader(table, index, True)