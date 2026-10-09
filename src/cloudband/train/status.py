"""Say where the experiment stands, from what it has left in the shared folder.

Nothing here trains or changes anything: it reads the manifests of finished runs, the
per-epoch state of runs in flight, the learning-rate search records and the locks, and
turns them into two tables and a short summary. It is safe to run while another session
is training.

A run is "done" when its manifest and checkpoint are both saved, "running" when a
session renewed its lock recently, "paused" when it has progress or a lock that nobody
renews any more (the next training cell continues it), and "not started" otherwise.
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from cloudband.train.loop import checkpoint_name
from cloudband.train.lr_search import proxy_protocol
from cloudband.train.protocol import (
    LR_SEARCH_GRID,
    PROXY_FROZEN_EPOCHS,
    PROXY_UNFROZEN_EPOCHS,
    TrainProtocol,
)
from cloudband.train.store import RunStore

# A lock renewed within this time belongs to a live session; the same lease as claim().
LEASE_SECONDS = 3600.0


@dataclass(frozen=True)
class Plan:
    """One model whose full runs are planned: a backbone of OCM, or the whole Swin."""

    label: str
    protocol: TrainProtocol
    suffix: str = ""


@dataclass(frozen=True)
class StatusReport:
    search: list[dict]
    runs: list[dict]
    summary: str


def _ago(seconds: float) -> str:
    if seconds < 90 * 60:
        return f"{max(seconds, 0) / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def _progress(path: Path) -> dict | None:
    """Epochs finished, best loss so far and learning rate of a run in flight."""
    if not path.is_file():
        return None
    try:
        try:
            # the file holds whole models; mapping it avoids reading the weights
            state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        except (RuntimeError, ValueError, OSError):
            state = torch.load(path, map_location="cpu", weights_only=False)
        losses = [float(value) for value in state["valid_losses"]]
        return {
            "epochs": len(losses),
            "best": min(losses) if losses else None,
            "learning_rate": state["fingerprint"]["learning_rate"],
        }
    except (OSError, EOFError, RuntimeError, KeyError, ValueError, pickle.UnpicklingError) as error:
        # a file being replaced or damaged must not stop a report
        return {"epochs": None, "best": None, "learning_rate": None, "error": str(error)}


def _lock(store: RunStore, name: str, now: float) -> dict | None:
    info = store.lock_info(name)
    if info is None:
        return None
    age = now - info["heartbeat"]
    mine = info["owner"] == store.session_id
    owner = f"{info['owner']}{' (this session)' if mine else ''}"
    return {"fresh": age < LEASE_SECONDS, "text": f"{owner}, {_ago(age)} ago"}


def _in_flight(store: RunStore, name: str, total: int, now: float) -> dict:
    """The row fields shared by a full run and a search candidate that is not done."""
    progress = _progress(store.epoch_state_path(name))
    lock = _lock(store, name, now)
    if lock is not None and lock["fresh"]:
        state = "running"
    elif progress is not None or lock is not None:
        state = "paused"
    else:
        state = "not started"
    done = progress["epochs"] if progress and progress["epochs"] is not None else 0
    note = ""
    if progress and progress.get("error"):
        note = f"progress file unreadable: {progress['error']}"
    elif state == "paused":
        note = f"continues from epoch {done}"
    return {
        "state": state,
        "epochs": f"{done}/{total}",
        "epochs_done": done,
        "best_val_loss": progress["best"] if progress else None,
        "lr": progress["learning_rate"] if progress else None,
        "session": lock["text"] if lock else "",
        "note": note,
    }


def run_row(
    store: RunStore,
    plan: Plan,
    seed: int,
    planned_lr: float | None = None,
    now: float | None = None,
) -> dict:
    now = time.time() if now is None else now
    name = checkpoint_name(plan.protocol, seed, plan.suffix)
    total = plan.protocol.frozen_epochs + plan.protocol.unfrozen_epochs
    row = {"model": plan.label, "seed": seed}
    if store.is_done(name):
        manifest = json.loads(store.manifest_path(name).read_text())
        config = manifest.get("config", {})
        history = config.get("valid_loss_history") or []
        convergence = store.convergence(name)
        row.update(
            state="done",
            epochs=f"{len(history)}/{total}",
            epochs_done=len(history),
            best_val_loss=convergence["best_val_loss"] if convergence else None,
            lr=config.get("learning_rate"),
            session="",
            note=(
                "best epoch is among the last 3: still improving"
                if convergence and convergence["still_improving"]
                else ""
            ),
        )
        return row
    row.update(_in_flight(store, name, total, now))
    if row["lr"] is None:
        row["lr"] = planned_lr
    return row


def search_rows(
    store: RunStore, protocol: TrainProtocol, now: float | None = None
) -> list[dict]:
    """One row per learning-rate candidate of an architecture."""
    now = time.time() if now is None else now
    seed = protocol.seeds[0]
    total = PROXY_FROZEN_EPOCHS + PROXY_UNFROZEN_EPOCHS

    found: dict[float, float] = {}
    partial = store.search_progress_path(protocol)
    if partial.is_file():
        found.update({r["learning_rate"]: r["val_loss"] for r in json.loads(partial.read_text())})
    final = store.load_search(protocol)
    grid = tuple(LR_SEARCH_GRID)
    if final is not None:
        grid = tuple(final.grid)
        found.update({run.learning_rate: run.val_loss for run in final.runs})
    winner = min(found, key=found.get) if final is not None and found else None

    rows = []
    for learning_rate in sorted(set(grid) | set(found)):
        row = {"model": protocol.architecture, "lr": learning_rate}
        if learning_rate in found:
            note = ""
            if learning_rate == winner:
                note = "winner" + (" (edge of the grid)" if final.winner_at_edge() else "")
            row.update(
                state="done", epochs=f"{total}/{total}", epochs_done=total,
                best_val_loss=found[learning_rate], session="", note=note,
            )
        else:
            candidate = proxy_protocol(protocol, learning_rate)
            name = checkpoint_name(candidate, seed, f"lrsearch-{learning_rate:g}")
            in_flight = _in_flight(store, name, total, now)
            in_flight.pop("lr")
            row.update(in_flight)
        rows.append(row)
    return rows


def _summary(search: list[dict], runs: list[dict], plans: list[Plan]) -> str:
    lines = ["Learning-rate search"]
    for architecture in dict.fromkeys(row["model"] for row in search):
        mine = [row for row in search if row["model"] == architecture]
        done = sum(row["state"] == "done" for row in mine)
        winner = next((row for row in mine if row["note"].startswith("winner")), None)
        tail = (
            f", winner {winner['lr']:g}{' (edge of the grid)' if 'edge' in winner['note'] else ''}"
            if winner else ""
        )
        lines.append(f"  {architecture}: {done} of {len(mine)} candidates done{tail}")

    lines.append("Full runs")
    for plan in plans:
        mine = [row for row in runs if row["model"] == plan.label]
        done = sum(row["state"] == "done" for row in mine)
        lines.append(f"  {plan.label}: {done} of {len(mine)} runs done")
    epochs_total = sum(
        plan.protocol.frozen_epochs + plan.protocol.unfrozen_epochs
        for plan in plans for _ in plan.protocol.seeds
    )
    epochs_done = sum(row["epochs_done"] for row in runs)
    lines.append(
        f"  epochs: {epochs_done} of {epochs_total} ({100 * epochs_done / epochs_total:.0f}%)"
    )

    every_model = [
        seed for seed in plans[0].protocol.seeds
        if all(
            row["state"] == "done"
            for row in runs if row["seed"] == seed
        )
    ]
    lines.append(f"  seeds finished for every model: {every_model or 'none yet'}")

    live = [
        f"{row['model']} seed {row['seed']} ({row['epochs']}, {row['session']})"
        for row in runs if row["state"] == "running"
    ] + [
        f"{row['model']} lr {row['lr']:g} ({row['epochs']}, {row['session']})"
        for row in search if row["state"] == "running"
    ]
    lines.append("Running now: " + ("; ".join(live) if live else "nothing"))
    paused = [
        f"{row['model']} seed {row['seed']} ({row['epochs']})"
        for row in runs if row["state"] == "paused"
    ] + [
        f"{row['model']} lr {row['lr']:g} ({row['epochs']})"
        for row in search if row["state"] == "paused"
    ]
    if paused:
        lines.append("Paused, to continue: " + "; ".join(paused))
    flagged = [row for row in runs if row["note"].startswith("best epoch")]
    if flagged:
        lines.append(f"Still improving at the end: {len(flagged)} finished run(s)")
    return "\n".join(lines)


def status_report(
    store: RunStore, plans: list[Plan], now: float | None = None
) -> StatusReport:
    """Read the shared folder and report the search, every planned run and a summary."""
    now = time.time() if now is None else now
    plans = list(plans)

    protocols = {plan.protocol.run_id: plan.protocol for plan in plans}
    search: list[dict] = []
    winners: dict[str, float] = {}
    for run_id, protocol in protocols.items():
        rows = search_rows(store, protocol, now)
        search.extend(rows)
        winner = next((r for r in rows if r["note"].startswith("winner")), None)
        if winner:
            winners[run_id] = winner["lr"]

    runs = [
        run_row(store, plan, seed, winners.get(plan.protocol.run_id), now)
        for plan in plans
        for seed in plan.protocol.seeds
    ]
    return StatusReport(search=search, runs=runs, summary=_summary(search, runs, plans))
