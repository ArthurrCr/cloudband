"""Read CloudSEN12+ samples with one HTTP request each, instead of GDAL's many.

Measured on the L1C archive: GDAL took about 4.1 s to read three bands of one
sample, and a single HTTP range request fetched the whole file in about 0.95 s.
Each file is only 1.2 to 1.5 MB (zstd, one strip per band), so there is no need to
fetch band by band: download the sample's bytes in one request, decode them in
memory, and keep the bands that are wanted.

The archive is a TACO v1 part file in which every sample sits at a byte offset.
tacoreader hands out paths of the form
``/vsisubfile/<offset>_<size>,/vsicurl/<url>``, and those are what is parsed here.
The image and its annotation usually sit side by side in the part file, and then a
single request covers both.
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests
from numpy.typing import NDArray

from cloudband.datasets.cloudsen12 import (
    ANNOTATION_INDEX,
    ID_COLUMN,
    IMAGE_INDEX,
    ROI_COLUMN,
    Sample,
    require_columns,
    valid_window,
)
from cloudband.pipelines.phase0 import RGN_INDICES

SUBFILE_PATTERN = re.compile(r"/vsisubfile/(\d+)_(\d+),(?:/vsicurl/)?(.+)")

# Two files closer than this are fetched in one request, reading the bytes between
# them and throwing them away, because a second request costs more than that.
MERGE_GAP_BYTES = 512 * 1024


@dataclass(frozen=True)
class Segment:
    """A byte range of a part file: where one file of the archive lives."""

    url: str
    offset: int
    size: int

    @property
    def end(self) -> int:
        """First byte after the segment."""
        return self.offset + self.size


def parse_subfile(path: str) -> Segment:
    """Read the offset, size and location out of a tacoreader path."""
    match = SUBFILE_PATTERN.match(str(path))
    if match is None:
        raise ValueError(
            f"unrecognised archive path {path!r}; expected "
            "'/vsisubfile/<offset>_<size>,/vsicurl/<url>'"
        )
    return Segment(url=match[3], offset=int(match[1]), size=int(match[2]))


Request = tuple[str, int, int, list[int]]


def plan_requests(segments: list[Segment]) -> list[Request]:
    """Group segments of one file that sit close together into single requests.

    Returns (url, start, end, member indices) per request, where end is exclusive and
    the members are positions in the input list.
    """
    order = sorted(
        range(len(segments)), key=lambda i: (segments[i].url, segments[i].offset)
    )
    groups: list[Request] = []
    for position in order:
        segment = segments[position]
        if groups:
            url, start, end, members = groups[-1]
            if url == segment.url and segment.offset - end <= MERGE_GAP_BYTES:
                groups[-1] = (url, start, max(end, segment.end), [*members, position])
                continue
        groups.append((segment.url, segment.offset, segment.end, [position]))
    return groups


class RangeFetcher:
    """Fetches byte ranges over HTTP (one connection per thread) or from a file."""

    def __init__(self, token: str | None = None, timeout: float = 120.0):
        self.token = token if token is not None else os.environ.get("HF_TOKEN")
        self.timeout = timeout
        self._local = threading.local()

    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            self._local.session = session
        return session

    def get(self, url: str, start: int, end: int) -> bytes:
        """Bytes [start, end) of the file at url."""
        if not url.startswith(("http://", "https://")):
            with open(url, "rb") as handle:
                handle.seek(start)
                return handle.read(end - start)

        headers = {"Range": f"bytes={start}-{end - 1}"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        response = self._session().get(url, headers=headers, timeout=self.timeout)
        # A server that ignores Range answers 200 with the whole archive; refusing
        # that is what keeps a misconfigured source from downloading gigabytes.
        if response.status_code != 206:
            raise OSError(
                f"HTTP {response.status_code} for bytes {start}-{end - 1} of {url}"
            )
        if len(response.content) != end - start:
            raise OSError(
                f"short read from {url}: got {len(response.content)} of "
                f"{end - start} bytes"
            )
        return response.content

    def fetch(self, segments: list[Segment]) -> list[bytes]:
        """The bytes of each segment, merging neighbours into single requests."""
        payloads: list[bytes] = [b""] * len(segments)
        for url, start, end, members in plan_requests(segments):
            blob = self.get(url, start, end)
            for position in members:
                segment = segments[position]
                first = segment.offset - start
                payloads[position] = blob[first:first + segment.size]
        return payloads


def decode_bands(data: bytes, bands: tuple[int, ...] | None) -> NDArray[np.integer]:
    """Decode a GeoTIFF held in memory, keeping the given zero-based bands."""
    from rasterio.io import MemoryFile

    with MemoryFile(data) as memory, memory.open() as source:
        if bands is None:
            stack: NDArray[np.integer] = source.read()
        else:
            stack = source.read([band + 1 for band in bands])
    return stack


def decode_first_band(data: bytes) -> NDArray[np.integer]:
    from rasterio.io import MemoryFile

    with MemoryFile(data) as memory, memory.open() as source:
        band: NDArray[np.integer] = source.read(1)
    return band


class FastReader:
    """A drop-in for cloudsen12.read_sample that fetches each sample in one request.

    bands are zero-based indices into the full stack. The default keeps red, green
    and near-infrared, and the image then has three channels; None keeps all of
    them, exactly like read_sample. Safe to call from many threads.
    """

    def __init__(
        self,
        bands: tuple[int, ...] | None = RGN_INDICES,
        fetcher: RangeFetcher | None = None,
    ):
        self.bands = bands
        self.fetcher = fetcher or RangeFetcher()

    def __call__(
        self, table: pd.DataFrame, index: int, crop_to_valid: bool = True
    ) -> Sample:
        require_columns(table, [ID_COLUMN])
        row = table.read(index)
        metadata = table.iloc[index]

        image_segment = parse_subfile(row.read(IMAGE_INDEX))
        annotation_segment = parse_subfile(row.read(ANNOTATION_INDEX))
        image_bytes, annotation_bytes = self.fetcher.fetch(
            [image_segment, annotation_segment]
        )

        image = decode_bands(image_bytes, self.bands)
        annotation = decode_first_band(annotation_bytes)
        if crop_to_valid:
            image = valid_window(image)
            annotation = valid_window(annotation)
        return Sample(
            identifier=str(metadata[ID_COLUMN]),
            image=image,
            annotation=annotation,
            roi_id=str(metadata[ROI_COLUMN]) if ROI_COLUMN in table.columns else None,
        )