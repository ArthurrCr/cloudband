"""Several training sessions at once on one big GPU.

The molab GPU has 96 GB, and one run of the comparison uses a few of them. The
scheduler already shares the work between sessions through locks on the Drive (that is
how several Colab notebooks train side by side), so the simplest way to fill the GPU is
to start more sessions: each worker here is a separate Python process that does what
the training cell of the notebook does, with its own session id and its own mirror of
the Drive folder. Separate processes keep each run's random state its own, so a run
gives the same result whichever worker trains it.

A worker starts from the notebook's mirror (hard links, no new download), reads the
cache the notebook built, and takes the micro-batch the notebook decided. Its output
goes to worker<k>.log in the work folder.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Worker:
    index: int
    process: subprocess.Popen
    log: Path

    @property
    def state(self) -> str:
        code = self.process.poll()
        if code is None:
            return "running"
        return "finished" if code == 0 else f"stopped with error (exit {code})"

    def stop(self) -> None:
        """Interrupt the worker as Ctrl+C would: the run in flight releases its lock and
        the worker sends its files to the Drive before exiting. Epoch progress is kept."""
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)

    def tail(self, lines: int = 12) -> str:
        try:
            with open(self.log, errors="replace", newline="") as handle:
                text = handle.read()
        except FileNotFoundError:
            return ""
        # progress bars rewrite their line with \r; keep what was last shown
        shown = [line.split("\r")[-1] for line in text.split("\n") if line.strip()]
        return "\n".join(shown[-lines:])


def launch(
    count: int,
    work_dir: Path,
    project_dir: Path,
    parent_training_dir: Path,
    token_file: Path,
    num_workers: int = 2,
    first_index: int = 1,
) -> list[Worker]:
    """Start `count` worker processes; they run until every unit is done or held."""
    work_dir, project_dir = Path(work_dir), Path(project_dir)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(project_dir / "src")] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    )
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    workers = []
    for index in range(first_index, first_index + count):
        log = work_dir / f"worker{index}.log"
        handle = open(log, "w")
        process = subprocess.Popen(
            [
                sys.executable, "-u", "-m", "cloudband.molab.workers",
                "--index", str(index),
                "--work-dir", str(work_dir),
                "--project-dir", str(project_dir),
                "--parent-training-dir", str(parent_training_dir),
                "--token-file", str(token_file),
                "--num-workers", str(num_workers),
            ],
            cwd=project_dir,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        workers.append(Worker(index, process, log))
    return workers


def gpu_summary() -> str:
    """Use and memory of the GPU, from nvidia-smi."""
    try:
        reply = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=20, check=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return f"nvidia-smi not available ({error})"
    use, used, total = (v.strip() for v in reply.stdout.splitlines()[0].split(","))
    return f"GPU in use {use}%, memory {int(used) / 1024:.1f} of {int(total) / 1024:.0f} GB"


def cpu_summary() -> str:
    try:
        load = os.getloadavg()[0]
    except OSError:
        return ""
    return f"CPU load (1 min) {load:.1f} on {os.cpu_count()} cores"


# ---- what one worker process does ----


def _load_tables(dataset):
    import pandas as pd

    def load_split(remote_source, split):
        return dataset.select(dataset.load_remote(remote_source), split=split)

    train = pd.concat(
        [load_split(dataset.REMOTE_L1C, "train"), load_split(dataset.REMOTE_L2A, "train")],
        ignore_index=True,
    )
    valid = pd.concat(
        [
            load_split(dataset.REMOTE_L1C, "validation"),
            load_split(dataset.REMOTE_L2A, "validation"),
        ],
        ignore_index=True,
    )
    return train, valid


def run_worker(args) -> None:
    from cloudband.datasets import cloudsen12 as dataset
    from cloudband.molab.drive_sync import (
        DriveMirror,
        DriveRunStore,
        drive_service,
        load_credentials,
    )
    from cloudband.train.cache import LocalCache
    from cloudband.train.data import build_dataloaders
    from cloudband.train.protocol import ocm_shared_protocol, swin_shared_protocol
    from cloudband.train.schedule import phase2_networks, run_everything

    work_dir, project_dir = Path(args.work_dir), Path(args.project_dir)
    credentials = load_credentials(Path(args.token_file).read_text())
    if credentials is None:
        raise SystemExit("the Google sign-in saved by the notebook no longer works")

    mirror = DriveMirror(
        drive_service(credentials), work_dir / f"worker{args.index}" / "cloudband_treino",
        log=lambda message: print(f"[worker {args.index}] {message}", flush=True),
    )
    print(f"[worker {args.index}] {mirror.seed_from(args.parent_training_dir)} file(s) "
          "taken from the notebook's copy", flush=True)
    mirror.start_background(interval_seconds=300)
    store = DriveRunStore(
        results_dir=mirror.local_dir / "results",
        checkpoint_dir=mirror.local_dir / "checkpoints",
        source_dir=project_dir / "models",
        mirror=mirror,
    )
    print(f"[worker {args.index}] session {store.session_id}", flush=True)

    decided = store.load_setting("micro_batch")
    if decided is None:
        raise SystemExit("no micro-batch decided yet: run the micro-batch cell first")

    train_table, valid_table = _load_tables(dataset)
    cache = LocalCache(work_dir / "cloudband_cache")
    train_reader = cache.reader("train", train_table)
    valid_reader = cache.reader("validation", valid_table)

    ocm_protocol = ocm_shared_protocol(learning_rate=1e-4)
    swin_protocol = swin_shared_protocol(learning_rate=1e-4)

    def make_dataloaders(micro_batch_size, protocol):
        return build_dataloaders(
            train_table=train_table,
            valid_table=valid_table,
            micro_batch_size=micro_batch_size,
            num_workers=args.num_workers,
            read_sample=train_reader,
            valid_read_sample=valid_reader,
            bands_selected=True,
            augmentation=protocol.augmentation if protocol.augment else None,
        )

    networks = phase2_networks(
        ocm_protocol,
        swin_protocol,
        make_dataloaders(decided["ocm"], ocm_protocol),
        make_dataloaders(decided["swin"], swin_protocol),
    )
    try:
        outcome = run_everything(networks, store, seeds=ocm_protocol.seeds)
        print(f"[worker {args.index}] done: trained {outcome.trained}", flush=True)
    finally:
        mirror.stop_background()
        report = mirror.push_all()
        print(f"[worker {args.index}] sent to the Drive: {len(report.uploaded)} file(s)",
              flush=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--parent-training-dir", required=True)
    parser.add_argument("--token-file", required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    run_worker(parser.parse_args(argv))


if __name__ == "__main__":
    main()
