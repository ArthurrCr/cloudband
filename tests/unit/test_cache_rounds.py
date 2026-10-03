import threading

import numpy as np
import pandas as pd
import pytest

from cloudband.datasets.cloudsen12 import Sample
from cloudband.train.cache import LocalCache, _error_label

PATCH = 24


def fake_read_sample(table, index, crop_to_valid=True):
    rng = np.random.default_rng(index)
    return Sample(
        identifier=f"scene-{index:03d}",
        image=rng.integers(0, 10000, (13, PATCH, PATCH)).astype(np.uint16),
        annotation=rng.integers(0, 4, (PATCH, PATCH)).astype(np.uint8),
    )


def make_table(n=6):
    return pd.DataFrame({"tortilla:id": [f"id{i}" for i in range(n)]})


class Recorder:
    def __init__(self):
        self.messages = []
        self.sleeps = []

    def progress(self, message):
        self.messages.append(message)

    def sleep(self, seconds):
        self.sleeps.append(seconds)

    def text(self):
        return "\n".join(self.messages)


class ThrottledUntil:
    """Every call fails until `calls` have been made, like a spent rate limit."""

    def __init__(self, calls, error=None):
        self.calls_needed = calls
        self.error = error or OSError("HTTP 429 for bytes 100-200 of https://x/part")
        self.count = 0
        self.lock = threading.Lock()

    def __call__(self, table, index, crop_to_valid=True):
        with self.lock:
            self.count += 1
            failing = self.count <= self.calls_needed
        if failing:
            raise self.error
        return fake_read_sample(table, index, crop_to_valid)


def build(tmp_path, reader, recorder, **kwargs):
    table = make_table()
    cache = LocalCache(tmp_path / "cache")
    options = dict(workers=4, retries=0, backoff_seconds=0, pause_seconds=30.0)
    options.update(kwargs)
    cache.build(
        "train", table, read_sample=reader, progress=recorder.progress,
        sleep=recorder.sleep, **options,
    )
    return cache, table


def test_samples_that_fail_in_a_wave_are_recovered_in_a_later_round(tmp_path):
    recorder = Recorder()
    reader = ThrottledUntil(calls=6)          # the whole first round is refused

    cache, table = build(tmp_path, reader, recorder)

    assert cache.reader("train", table) is not None
    assert "round 2" in recorder.text()
    assert recorder.sleeps == [30.0]          # one pause, before round two


def test_each_round_uses_half_the_threads_of_the_one_before(tmp_path):
    recorder = Recorder()
    reader = ThrottledUntil(calls=12)         # rounds one and two are both refused

    build(tmp_path, reader, recorder, workers=8)

    text = recorder.text()
    assert "copying 6 of 6 samples, 8 threads" in text
    assert "round 2, retrying 6 samples, 4 threads" in text
    assert "round 3, retrying 6 samples, 2 threads" in text


def test_the_final_error_names_the_most_common_cause(tmp_path):
    recorder = Recorder()
    reader = ThrottledUntil(calls=10**6)      # never recovers

    with pytest.raises(RuntimeError) as caught:
        build(tmp_path, reader, recorder)

    message = str(caught.value)
    assert "6 samples failed" in message
    assert "3 rounds" in message
    assert "most common errors" in message
    assert "6x OSError: HTTP # for bytes #-# of https://x/part" in message


def test_the_progress_line_counts_the_failures(tmp_path):
    recorder = Recorder()

    with pytest.raises(RuntimeError):
        build(tmp_path, ThrottledUntil(calls=10**6), recorder, rounds=1)

    assert "failed so far" in recorder.text()


def test_a_single_round_gives_up_without_pausing(tmp_path):
    recorder = Recorder()

    with pytest.raises(RuntimeError, match="after 0 retries and 1 rounds"):
        build(tmp_path, ThrottledUntil(calls=10**6), recorder, rounds=1)

    assert recorder.sleeps == []


def test_different_errors_are_counted_separately(tmp_path):
    table = make_table()
    cache = LocalCache(tmp_path / "cache")

    def mixed(table_, index, crop_to_valid=True):
        if index < 4:
            raise OSError("HTTP 429 for bytes 1-2 of u")
        raise ValueError("bad decode")

    with pytest.raises(RuntimeError) as caught:
        cache.build(
            "train", table, read_sample=mixed, progress=lambda _: None,
            sleep=lambda _: None, workers=2, retries=0, rounds=1,
        )

    message = str(caught.value)
    assert "4x OSError: HTTP # for bytes #-# of u" in message
    assert "2x ValueError: bad decode" in message


def test_numbers_are_masked_so_equal_errors_group():
    a = _error_label(OSError("HTTP 429 for bytes 100-200"))
    b = _error_label(OSError("HTTP 429 for bytes 9000-9100"))

    assert a == b == "OSError: HTTP # for bytes #-#"