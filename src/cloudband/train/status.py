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
import math
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
    # the protocol the learning-rate search of this model runs under; its own files
    # and candidates. None means the search is the one of `protocol` itself.
    search_protocol: TrainProtocol | None = None

    @property
    def search(self) -> TrainProtocol:
        return self.search_protocol or self.protocol


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
    try:
        if not path.is_file():
            return None
    except OSError:
        return None          # an entry the Drive still lists after it was trashed
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


def _is_done(store: RunStore, name: str) -> bool:
    """store.is_done for a report: an entry the Drive cannot read counts as not done."""
    try:
        return store.is_done(name)
    except OSError:
        return False


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
    if _is_done(store, name):
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
        return _mark_rate(row, name, planned_lr)
    row.update(_in_flight(store, name, total, now))
    if row["lr"] is None:
        row["lr"] = planned_lr
    return _mark_rate(row, name, planned_lr)


def _mark_rate(row: dict, name: str, chosen_lr: float | None) -> dict:
    """Say which rate a run was trained at and whether it is the one the search chose.

    A run trained at another rate cannot be continued or kept: the runs of one model
    must all share the rate its search chose.
    """
    trained_lr = row["lr"] if row["state"] != "not started" else None
    other = (
        trained_lr is not None
        and chosen_lr is not None
        and not math.isclose(trained_lr, chosen_lr, rel_tol=1e-9)
    )
    row.update(name=name, chosen_lr=chosen_lr, at_other_rate=other)
    if other:
        row["note"] = (
            f"trained at {trained_lr:g} but the search chose {chosen_lr:g}: "
            "discard it before going on"
        )
    return row


def search_rows(
    store: RunStore, protocol: TrainProtocol, label: str, now: float | None = None
) -> list[dict]:
    """One row per learning-rate candidate of a model's search."""
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
        row = {"model": label, "lr": learning_rate}
        if learning_rate in found:
            note = "diverged (infinite loss)" if found[learning_rate] == float("inf") else ""
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


def finished_names(store: RunStore, plans: list[Plan], now: float | None = None) -> set[str]:
    """Names of the runs and search candidates that are over.

    Their progress files and locks are of no use any more: a finished run is its manifest
    and checkpoint, and a finished candidate is its record in the search plus its
    validation curve. A candidate whose curve is not saved yet is left out, so that the
    progress file that still holds the curve is not thrown away.
    """
    names: set[str] = set()
    for plan in plans:
        for seed in plan.protocol.seeds:
            name = checkpoint_name(plan.protocol, seed, plan.suffix)
            if _is_done(store, name):
                names.add(name)
        curves = store.load_curves(plan.search)
        for row in search_rows(store, plan.search, plan.label, now):
            diverged = row["best_val_loss"] == float("inf")
            if row["state"] == "done" and (diverged or row["lr"] in curves):
                candidate = proxy_protocol(plan.search, row["lr"])
                names.add(
                    checkpoint_name(candidate, plan.search.seeds[0], f"lrsearch-{row['lr']:g}")
                )
    return names


def _summary(search: list[dict], runs: list[dict], plans: list[Plan]) -> str:
    lines = ["Learning-rate search"]
    for model in dict.fromkeys(row["model"] for row in search):
        mine = [row for row in search if row["model"] == model]
        done = sum(row["state"] == "done" for row in mine)
        winner = next((row for row in mine if row["note"].startswith("winner")), None)
        tail = (
            f", winner {winner['lr']:g}{' (edge of the grid)' if 'edge' in winner['note'] else ''}"
            if winner else ""
        )
        lines.append(f"  {model}: {done} of {len(mine)} candidates done{tail}")

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
    other = [row for row in runs if row["at_other_rate"]]
    if other:
        lines.append(
            f"At another rate than the chosen one: {len(other)} run(s), to discard before "
            "going on: " + "; ".join(f"{row['model']} seed {row['seed']}" for row in other)
        )
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

    searches = {plan.search.run_id: (plan.label, plan.search) for plan in plans}
    search: list[dict] = []
    winners: dict[str, float] = {}
    for run_id, (label, protocol) in searches.items():
        rows = search_rows(store, protocol, label, now)
        search.extend(rows)
        winner = next((r for r in rows if r["note"].startswith("winner")), None)
        if winner:
            winners[run_id] = winner["lr"]

    runs = [
        run_row(store, plan, seed, winners.get(plan.search.run_id), now)
        for plan in plans
        for seed in plan.protocol.seeds
    ]
    return StatusReport(search=search, runs=runs, summary=_summary(search, runs, plans))
