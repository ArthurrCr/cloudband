import re
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pandas as pd
import pytest
from rasterio.io import MemoryFile

from cloudband.datasets import cloudsen12 as dataset
from cloudband.datasets import fast_read
from cloudband.datasets.fast_read import (
    MERGE_GAP_BYTES,
    FastReader,
    RangeFetcher,
    Segment,
    parse_subfile,
    plan_requests,
)
from cloudband.pipelines.phase0 import select_rgn
from cloudband.train.cache import LocalCache

CANVAS = 512


def tiff_bytes(array, **profile):
    count = 1 if array.ndim == 2 else array.shape[0]
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff",
            width=CANVAS,
            height=CANVAS,
            count=count,
            dtype=str(array.dtype),
            compress="zstd",
            interleave="band",
            **profile,
        ) as target:
            target.write(array if array.ndim == 3 else array[None])
        return memory.read()


class Archive:
    """A fake TACO part file: image and annotation per sample, at known offsets."""

    def __init__(self, n=4, gap=0, seed=0):
        rng = np.random.default_rng(seed)
        self.blob = bytearray(b"HEADER-JUNK" * 100)
        self.entries = []
        for _ in range(n):
            image = rng.integers(0, 60, (13, CANVAS, CANVAS)).astype(np.uint16)
            annotation = rng.integers(0, 4, (CANVAS, CANVAS)).astype(np.uint8)
            annotation[:, :3] = 99
            image_bytes, annotation_bytes = tiff_bytes(image), tiff_bytes(annotation)
            image_at = len(self.blob)
            self.blob += image_bytes
            self.blob += b"\0" * gap
            annotation_at = len(self.blob)
            self.blob += annotation_bytes
            self.blob += b"PADDING" * 50
            self.entries.append(
                (image_at, len(image_bytes), annotation_at, len(annotation_bytes))
            )
        self.blob = bytes(self.blob)

    def table(self, location):
        """A table whose archive paths point at location (a URL or a file path)."""
        frame = TaxoFrame(
            {
                "tortilla:id": [f"patch-{i}" for i in range(len(self.entries))],
                "roi_id": [f"roi-{i % 2}" for i in range(len(self.entries))],
                "image_path": [
                    self._path(location, e[0], e[1]) for e in self.entries
                ],
                "label_path": [
                    self._path(location, e[2], e[3]) for e in self.entries
                ],
            }
        )
        return frame

    @staticmethod
    def _path(location, offset, size):
        if location.startswith("http"):
            return f"/vsisubfile/{offset}_{size},/vsicurl/{location}"
        return f"/vsisubfile/{offset}_{size},{location}"


class FakeRow:
    def __init__(self, paths):
        self.paths = paths

    def read(self, index):
        return self.paths[index]


class TaxoFrame(pd.DataFrame):
    """A DataFrame with tacoreader's read(), enough for the readers under test."""

    @property
    def _constructor(self):
        return TaxoFrame

    def read(self, index):
        row = self.iloc[index]
        return FakeRow([row["image_path"], row["label_path"]])


class Server(ThreadingHTTPServer):
    daemon_threads = True


def make_handler():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            server = self.server
            header = self.headers.get("Range")
            auth = self.headers.get("Authorization")
            server.log.append({"range": header, "auth": auth})
            data = server.blob
            if server.rate_limited > 0:
                server.rate_limited -= 1
                self.send_response(429)
                for name, value in server.rate_limit_headers.items():
                    self.send_header(name, value)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if server.mode == "ignore_range" or header is None:
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            span = re.match(r"bytes=(\d+)-(\d+)", header)
            start, end = int(span[1]), int(span[2])
            chunk = data[start:end + 1]
            if server.mode == "short":
                chunk = chunk[:-5]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
            self.send_header("Content-Length", str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)

        def log_message(self, *args):
            return None

    return Handler


@pytest.fixture
def serve(tmp_path):
    started = []

    def start(archive):
        server = Server(("127.0.0.1", 0), make_handler())
        server.blob, server.log, server.mode = archive.blob, [], "normal"
        server.rate_limited, server.rate_limit_headers = 0, {}
        threading.Thread(target=server.serve_forever, daemon=True).start()
        started.append(server)
        url = f"http://127.0.0.1:{server.server_address[1]}/part.taco"
        local = tmp_path / "part.taco"
        local.write_bytes(archive.blob)
        return server, url, str(local)

    yield start
    for server in started:
        server.shutdown()
        server.server_close()


def same_sample(a, b):
    assert a.identifier == b.identifier
    assert a.roi_id == b.roi_id
    assert np.array_equal(a.image, b.image)
    assert a.image.dtype == b.image.dtype
    assert np.array_equal(a.annotation, b.annotation)
    assert a.annotation.dtype == b.annotation.dtype


def test_it_returns_the_same_sample_as_the_original_reader(serve):
    archive = Archive()
    _, url, local = serve(archive)
    remote, on_disk = archive.table(url), archive.table(local)

    for index in range(len(remote)):
        original = dataset.read_sample(on_disk, index)
        fast = FastReader(bands=None)(remote, index)
        same_sample(fast, original)
        assert fast.image.shape == (13, 509, 509)


def test_the_default_keeps_red_green_and_nir_in_that_order(serve):
    archive = Archive()
    _, url, local = serve(archive)
    original = dataset.read_sample(archive.table(local), 1)

    fast = FastReader()(archive.table(url), 1)

    assert fast.image.shape == (3, 509, 509)
    assert np.array_equal(fast.image, select_rgn(original.image))


def test_cropping_can_be_turned_off(serve):
    archive = Archive(n=1)
    _, url, local = serve(archive)

    fast = FastReader(bands=None)(archive.table(url), 0, crop_to_valid=False)
    original = dataset.read_sample(archive.table(local), 0, crop_to_valid=False)

    assert fast.image.shape == (13, 512, 512)
    same_sample(fast, original)


def test_adjacent_image_and_annotation_take_a_single_request(serve):
    archive = Archive(n=3)
    server, url, _ = serve(archive)
    table = archive.table(url)

    for index in range(3):
        FastReader()(table, index)

    assert len(server.log) == 3


def test_distant_files_take_two_requests(serve):
    archive = Archive(n=2, gap=MERGE_GAP_BYTES + 1000)
    server, url, _ = serve(archive)

    FastReader()(archive.table(url), 0)

    assert len(server.log) == 2


def test_a_server_that_ignores_range_is_refused(serve):
    archive = Archive(n=1)
    server, url, _ = serve(archive)
    server.mode = "ignore_range"

    with pytest.raises(OSError, match="HTTP 200"):
        FastReader()(archive.table(url), 0)


def test_a_short_read_is_refused(serve):
    archive = Archive(n=1)
    server, url, _ = serve(archive)
    server.mode = "short"

    with pytest.raises(OSError, match="short read"):
        FastReader()(archive.table(url), 0)


def test_the_token_is_sent_when_given_and_not_otherwise(serve, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    archive = Archive(n=1)
    server, url, _ = serve(archive)
    table = archive.table(url)

    FastReader(fetcher=RangeFetcher(token="secret"))(table, 0)
    FastReader(fetcher=RangeFetcher())(table, 0)

    assert server.log[0]["auth"] == "Bearer secret"
    assert server.log[1]["auth"] is None


def test_the_token_falls_back_to_the_environment(serve, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "from-env")
    archive = Archive(n=1)
    server, url, _ = serve(archive)

    FastReader()(archive.table(url), 0)

    assert server.log[0]["auth"] == "Bearer from-env"


def test_a_local_part_file_is_read_without_http(serve):
    archive = Archive(n=1)
    _, _, local = serve(archive)
    table = archive.table(local)

    fast = FastReader(bands=None)(table, 0)

    same_sample(fast, dataset.read_sample(table, 0))


def test_many_threads_get_the_same_samples_as_one(serve):
    archive = Archive(n=6)
    _, url, _ = serve(archive)
    table = archive.table(url)
    reader = FastReader()

    serial = [reader(table, i) for i in range(6)]
    with ThreadPoolExecutor(max_workers=6) as pool:
        parallel = list(pool.map(lambda i: reader(table, i), range(6)))

    for a, b in zip(serial, parallel):
        same_sample(a, b)


def test_parse_subfile_reads_offset_size_and_location():
    path = "/vsisubfile/457106709_1457907,/vsicurl/https://huggingface.co/x/part.taco"

    assert parse_subfile(path) == Segment(
        "https://huggingface.co/x/part.taco", 457106709, 1457907
    )
    assert parse_subfile("/vsisubfile/5_6,/data/part.taco").url == "/data/part.taco"


def test_an_unrecognised_path_says_what_it_expected():
    with pytest.raises(ValueError, match="unrecognised archive path"):
        parse_subfile("/vsicurl/https://example.org/file.tif")


def test_requests_merge_only_close_segments_of_the_same_file():
    close = [Segment("a", 0, 100), Segment("a", 150, 100)]
    far = [Segment("a", 0, 100), Segment("a", 100 + MERGE_GAP_BYTES + 1, 100)]
    other_file = [Segment("a", 0, 100), Segment("b", 100, 100)]

    assert len(plan_requests(close)) == 1
    assert len(plan_requests(far)) == 2
    assert len(plan_requests(other_file)) == 2
    assert plan_requests(close)[0][1:3] == (0, 250)


def test_the_cache_stores_the_three_bands_straight_from_the_fast_reader(
    serve, tmp_path
):
    archive = Archive(n=4)
    _, url, local = serve(archive)
    remote, on_disk = archive.table(url), archive.table(local)
    cache = LocalCache(tmp_path / "cache")

    cache.build(
        "train", remote, read_sample=FastReader(), workers=3,
        progress=lambda _: None, bands_selected=True,
    )

    reader = cache.reader("train", remote)
    for index in range(len(remote)):
        cached = reader(remote, index)
        original = dataset.read_sample(on_disk, index)
        assert np.array_equal(cached.image, select_rgn(original.image))
        assert np.array_equal(cached.annotation, original.annotation)
        assert cached.identifier == original.identifier


def test_the_module_exposes_what_the_notebook_imports():
    assert callable(fast_read.FastReader)
    assert fast_read.MERGE_GAP_BYTES > 0


class Sleeps:
    """Stands in for time.sleep, recording the waits instead of waiting."""

    def __init__(self):
        self.waits = []

    def __call__(self, seconds):
        self.waits.append(seconds)


def test_a_429_is_waited_out_using_retry_after_and_then_succeeds(serve):
    archive = Archive(n=1)
    server, url, _ = serve(archive)
    server.rate_limited, server.rate_limit_headers = 2, {"Retry-After": "7"}
    sleeps = Sleeps()

    sample = FastReader(fetcher=RangeFetcher(sleep=sleeps))(archive.table(url), 0)

    assert sample.image.shape == (3, 509, 509)
    assert len(sleeps.waits) == 2
    assert all(7.0 <= wait <= 9.0 for wait in sleeps.waits)
    assert len(server.log) == 3                  # two refused, one served


def test_the_ratelimit_header_is_used_when_there_is_no_retry_after(serve):
    archive = Archive(n=1)
    server, url, _ = serve(archive)
    server.rate_limited = 1
    server.rate_limit_headers = {"RateLimit": '"api";r=0;t=41'}
    sleeps = Sleeps()

    FastReader(fetcher=RangeFetcher(sleep=sleeps))(archive.table(url), 0)

    assert 41.0 <= sleeps.waits[0] <= 43.0


def test_a_429_without_any_hint_waits_a_default_while(serve):
    archive = Archive(n=1)
    server, url, _ = serve(archive)
    server.rate_limited = 1
    sleeps = Sleeps()

    FastReader(fetcher=RangeFetcher(sleep=sleeps))(archive.table(url), 0)

    assert fast_read.DEFAULT_RATE_LIMIT_WAIT <= sleeps.waits[0] <= (
        fast_read.DEFAULT_RATE_LIMIT_WAIT + 2.0
    )


def test_an_absurd_wait_is_capped_at_one_window(serve):
    archive = Archive(n=1)
    server, url, _ = serve(archive)
    server.rate_limited, server.rate_limit_headers = 1, {"Retry-After": "86400"}
    sleeps = Sleeps()

    FastReader(fetcher=RangeFetcher(sleep=sleeps))(archive.table(url), 0)

    assert sleeps.waits[0] <= fast_read.MAX_RATE_LIMIT_WAIT + 2.0


def test_it_gives_up_after_the_retries_and_says_it_was_rate_limited(serve):
    archive = Archive(n=1)
    server, url, _ = serve(archive)
    server.rate_limited, server.rate_limit_headers = 99, {"Retry-After": "1"}
    sleeps = Sleeps()
    fetcher = RangeFetcher(max_rate_limit_retries=3, sleep=sleeps)

    with pytest.raises(OSError, match="HTTP 429"):
        FastReader(fetcher=fetcher)(archive.table(url), 0)

    assert len(sleeps.waits) == 3
    assert len(server.log) == 4                  # the first try and three retries