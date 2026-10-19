"""Remove the stale copies a mounted Drive leaves behind when a file is replaced.

Replacing a file on the Drive mounted in Colab does not overwrite it: the new version is
a new file with the same name, and the old one stays, either in the same folder or loose
in the root of My Drive. Every epoch of a run therefore adds a copy of its state (half a
gigabyte for convnext), charged to the account that wrote it. This module finds those
copies through the Drive API and sends the superseded ones to the trash.

Rules, in order of importance:
- the newest complete copy of every name is always kept, wherever it is, unless the name
  belongs to a run or a search candidate that is over: its progress and lock files are of
  no use any more and every copy goes (the validation curve of a candidate is saved
  first, see salvage_curves);
- an interrupted copy (".partial") older than an hour is of no use either;
- a copy in the shared folders is preferred to one loose in the root of My Drive;
- a copy much smaller than the largest one of its name (a write cut short) is never the
  one that is kept, and never the reason another one is removed;
- only files this account owns are trashed (the others are reported, for their owner's
  account to clean up), and trashing can be undone: the trash is never emptied here.

The planning is plain data, so it is tested without any network.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

FOLDER = "application/vnd.google-apps.folder"
SHORTCUT = "application/vnd.google-apps.shortcut"
# a copy smaller than this share of the largest of its name is taken as cut short
COMPLETE_SHARE = 0.9
ROOT = "root"
SHARED = "shared"
# progress files and locks: their names are the run's name plus one of these
PROGRESS_SUFFIXES = (".state.pt", ".lock")
RECENT_SECONDS = 3600.0


@dataclass(frozen=True)
class Copy:
    id: str
    name: str
    size: int
    modified: str      # RFC 3339, so that strings sort in time order
    mine: bool
    where: str         # ROOT (loose in My Drive) or SHARED (inside the project folders)


@dataclass
class TidyPlan:
    keep: list[Copy] = field(default_factory=list)
    trash: list[Copy] = field(default_factory=list)       # superseded and owned here
    others: list[Copy] = field(default_factory=list)      # superseded, owned elsewhere
    last_in_root: list[Copy] = field(default_factory=list)  # kept, but loose in My Drive
    of_finished: list[Copy] = field(default_factory=list)   # progress of runs that are over

    @property
    def trash_bytes(self) -> int:
        return sum(copy.size for copy in self.trash)

    @property
    def others_bytes(self) -> int:
        return sum(copy.size for copy in self.others)


def _age_seconds(modified: str, now: datetime) -> float:
    stamp = datetime.fromisoformat(modified.replace("Z", "+00:00"))
    return (now - stamp).total_seconds()


def _run_name(name: str) -> str | None:
    for suffix in PROGRESS_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def plan_tidy(
    copies: list[Copy], finished: frozenset[str] | set[str] = frozenset(),
    now: datetime | None = None,
) -> TidyPlan:
    """Decide which copy of every name stays and which are superseded.

    `finished` holds the names of runs and search candidates that are over (see
    status.finished_names): every progress file and lock of those is superseded.
    """
    now = now or datetime.now(timezone.utc)
    groups: dict[str, list[Copy]] = {}
    for copy in copies:
        groups.setdefault(copy.name, []).append(copy)

    plan = TidyPlan()
    for name, group in groups.items():
        over = _run_name(name) in finished
        interrupted = name.endswith(".partial") and all(
            _age_seconds(c.modified, now) >= RECENT_SECONDS for c in group
        )
        if over or interrupted:
            kept = None
            if over:
                plan.of_finished.extend(group)
        else:
            largest = max(copy.size for copy in group)
            complete = [c for c in group if c.size >= COMPLETE_SHARE * largest]
            inside = [c for c in complete if c.where == SHARED]
            kept = max(inside or complete, key=lambda c: c.modified)
            plan.keep.append(kept)
            if kept.where == ROOT:
                plan.last_in_root.append(kept)
        for copy in group:
            if kept is not None and copy.id == kept.id:
                continue
            (plan.trash if copy.mine else plan.others).append(copy)
    return plan


# ---- the Drive API, kept thin ----

def drive_service():
    """The Drive API as the account signed in to this Colab session."""
    from google.colab import auth
    from googleapiclient.discovery import build

    auth.authenticate_user()
    return build("drive", "v3", cache_discovery=False)


def _quote(name: str) -> str:
    return name.replace("\\", "\\\\").replace("'", "\\'")


def _pages(service, query: str, fields: str):
    token = None
    while True:
        reply = (
            service.files()
            .list(q=query, fields=f"nextPageToken, files({fields})", pageSize=200,
                  pageToken=token)
            .execute()
        )
        yield from reply.get("files", [])
        token = reply.get("nextPageToken")
        if not token:
            return


def resolve_folder(service, parts: tuple[str, ...]) -> str:
    """The id of a folder, walking down from My Drive; a shortcut leads to its target."""
    parent = ROOT
    for part in parts:
        query = (
            f"name = '{_quote(part)}' and '{parent}' in parents and trashed = false "
            f"and (mimeType = '{FOLDER}' or mimeType = '{SHORTCUT}')"
        )
        found = list(_pages(service, query, "id, mimeType, shortcutDetails"))
        if not found:
            raise FileNotFoundError(f"folder {part!r} not found in the Drive of this account")
        entry = found[0]
        parent = (
            entry["shortcutDetails"]["targetId"] if entry["mimeType"] == SHORTCUT else entry["id"]
        )
    return parent


def list_copies(service, folders: dict[str, str], prefixes: tuple[str, ...]) -> list[Copy]:
    """Every file of the project in the given folders and loose in the root of My Drive.

    `folders` maps a folder id to ROOT or SHARED. Only names that start with one of the
    prefixes (the run ids) are listed, so nothing else in the Drive is ever touched.
    """
    copies = []
    fields = "id, name, size, modifiedTime, ownedByMe"
    for folder_id, where in folders.items():
        query = f"'{folder_id}' in parents and trashed = false and mimeType != '{FOLDER}'"
        for entry in _pages(service, query, fields):
            if entry["name"].startswith(prefixes):
                copies.append(
                    Copy(
                        id=entry["id"],
                        name=entry["name"],
                        size=int(entry.get("size", 0)),
                        modified=entry["modifiedTime"],
                        mine=bool(entry.get("ownedByMe")),
                        where=where,
                    )
                )
    return copies


def trash(service, copies: list[Copy]) -> int:
    """Send copies to the trash (reversible). Returns how many were trashed."""
    for copy in copies:
        service.files().update(fileId=copy.id, body={"trashed": True}).execute()
    return len(copies)


def folder_parts(path: str, mount: str = "/content/drive/MyDrive") -> tuple[str, ...]:
    """The folders between My Drive and a path on the mounted Drive."""
    return PurePosixPath(path).relative_to(mount).parts


def _read_valid_losses(service, file_id: str) -> list[float]:
    """Download a progress file and read the validation losses it holds."""
    import torch
    from googleapiclient.http import MediaIoBaseDownload

    with tempfile.TemporaryDirectory() as scratch:
        target = Path(scratch) / "state.pt"
        with open(target, "wb") as handle:
            downloader = MediaIoBaseDownload(
                handle, service.files().get_media(fileId=file_id), chunksize=64 * 1024 * 1024
            )
            done = False
            while not done:
                _, done = downloader.next_chunk()
        state = torch.load(target, map_location="cpu", weights_only=False, mmap=True)
        return [float(value) for value in state["valid_losses"]]


def salvage_curves(
    service, store, plans, copies: list[Copy], read=_read_valid_losses
) -> tuple[list[str], list[str]]:
    """Save the validation curve of every finished candidate that has none saved.

    The search record keeps only the best loss of a candidate, and the curve lived in
    the progress file that is cleared when the candidate ends, leaving only the stale
    copies on Drive. The newest complete copy holds the whole curve; it is kept only
    when it has every epoch of the proxy budget, so a partial curve is never saved as
    if it were whole. Returns what was saved and what could not be.
    """
    from cloudband.train.loop import checkpoint_name
    from cloudband.train.lr_search import proxy_protocol
    from cloudband.train.protocol import PROXY_FROZEN_EPOCHS, PROXY_UNFROZEN_EPOCHS
    from cloudband.train.status import search_rows

    epochs = PROXY_FROZEN_EPOCHS + PROXY_UNFROZEN_EPOCHS
    saved, missing = [], []
    done_searches = set()
    for plan in plans:
        protocol = plan.search
        if protocol.run_id in done_searches:
            continue
        done_searches.add(protocol.run_id)
        have = store.load_curves(protocol)
        for row in search_rows(store, protocol, plan.label):
            rate = row["lr"]
            if row["state"] != "done" or rate in have or row["best_val_loss"] == float("inf"):
                continue
            name = checkpoint_name(
                proxy_protocol(protocol, rate), protocol.seeds[0], f"lrsearch-{rate:g}"
            )
            label = f"{plan.label} lr {rate:g}"
            candidates = [c for c in copies if c.name == f"{name}.state.pt"]
            if not candidates:
                missing.append(f"{label}: no progress file left")
                continue
            largest = max(c.size for c in candidates)
            newest = max(
                (c for c in candidates if c.size >= COMPLETE_SHARE * largest),
                key=lambda c: c.modified,
            )
            curve = read(service, newest.id)
            if len(curve) != epochs:
                missing.append(f"{label}: the newest copy has {len(curve)} of {epochs} epochs")
                continue
            store.save_curve(protocol, rate, curve)
            saved.append(label)
    return saved, missing
