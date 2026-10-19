"""Remove the stale copies a mounted Drive leaves behind when a file is replaced.

Replacing a file on the Drive mounted in Colab does not overwrite it: the new version is
a new file with the same name, and the old one stays, either in the same folder or loose
in the root of My Drive. Every epoch of a run therefore adds a copy of its state (half a
gigabyte for convnext), charged to the account that wrote it. This module finds those
copies through the Drive API and sends the superseded ones to the trash.

Rules, in order of importance:
- the newest complete copy of every name is always kept, wherever it is;
- a copy in the shared folders is preferred to one loose in the root of My Drive;
- a copy much smaller than the largest one of its name (a write cut short) is never the
  one that is kept, and never the reason another one is removed;
- only files this account owns are trashed (the others are reported, for their owner's
  account to clean up), and trashing can be undone: the trash is never emptied here.

The planning is plain data, so it is tested without any network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath

FOLDER = "application/vnd.google-apps.folder"
SHORTCUT = "application/vnd.google-apps.shortcut"
# a copy smaller than this share of the largest of its name is taken as cut short
COMPLETE_SHARE = 0.9
ROOT = "root"
SHARED = "shared"


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

    @property
    def trash_bytes(self) -> int:
        return sum(copy.size for copy in self.trash)

    @property
    def others_bytes(self) -> int:
        return sum(copy.size for copy in self.others)


def plan_tidy(copies: list[Copy]) -> TidyPlan:
    """Decide which copy of every name stays and which are superseded."""
    groups: dict[str, list[Copy]] = {}
    for copy in copies:
        groups.setdefault(copy.name, []).append(copy)

    plan = TidyPlan()
    for group in groups.values():
        largest = max(copy.size for copy in group)
        complete = [c for c in group if c.size >= COMPLETE_SHARE * largest]
        inside = [c for c in complete if c.where == SHARED]
        kept = max(inside or complete, key=lambda c: c.modified)
        plan.keep.append(kept)
        if kept.where == ROOT:
            plan.last_in_root.append(kept)
        for copy in group:
            if copy.id == kept.id:
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
