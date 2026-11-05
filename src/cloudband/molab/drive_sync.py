"""The Colab shared folder on Google Drive, from a molab session.

Colab mounts My Drive at /content/drive, so RunStore writes straight into the shared
folder. molab has no mount, and its disk is lost when the session ends. Here the
session works in a local folder that mirrors the same Drive folder through the Drive
API, signed in with the user's own Google account (OAuth), so Colab and molab read and
write the same runs:

- `restore` downloads `results/` and `checkpoints/` of the Drive folder once, at the
  start, so the runs already done on Colab are found and resumed;
- every `push` sends up what changed since (new or changed files, by size and
  modification time), and sends to the Drive trash what was removed locally, for
  instance by RunStore.discard. The trash can be undone, nothing is deleted for good.

A file still being written (a `.partial`, or one modified in the last `settle_seconds`)
waits for the next push. A push refuses to run before a successful restore, so an empty
local folder is never taken as "everything was deleted".

When the Drive holds several copies of a name in one folder (what the Colab mount leaves
behind when it replaces a file), the newest complete one is used, the same rule as
cloudband.colab.drive_tidy, which removes the stale copies.

Colab and molab must not train the same run at the same moment: locks reach the other
side only at the next push.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from cloudband.colab.drive_tidy import COMPLETE_SHARE, FOLDER, SHORTCUT, _pages, _quote

DRIVE_SCOPE = "https://www.googleapis.com/auth/drive"
MIRRORED = ("results", "checkpoints")
SKIPPED_SUFFIXES = (".partial",)


@dataclass(frozen=True)
class PushReport:
    uploaded: tuple[str, ...]
    deleted: tuple[str, ...]
    waiting: tuple[str, ...]

    @property
    def changed(self) -> bool:
        return bool(self.uploaded or self.deleted)


def _skipped(name: str) -> bool:
    return name.endswith(SKIPPED_SUFFIXES) or ".partial." in name


def _signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def pick_copy(copies: list[dict]) -> dict:
    """The newest copy among those not cut short (close to the largest in size)."""
    largest = max(int(c.get("size", 0)) for c in copies)
    complete = [c for c in copies if int(c.get("size", 0)) >= COMPLETE_SHARE * largest]
    return max(complete, key=lambda c: c["modifiedTime"])


def local_files(root: Path, mirrored: tuple[str, ...] = MIRRORED) -> dict[str, Path]:
    files = {}
    for top in mirrored:
        base = root / top
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and not _skipped(path.name):
                files[path.relative_to(root).as_posix()] = path
    return files


def resolve_path(service, parts: tuple[str, ...]) -> str:
    """The id of a folder below My Drive, following shortcuts, as on the Colab mount."""
    parent = "root"
    for part in parts:
        query = (
            f"name = '{_quote(part)}' and '{parent}' in parents and trashed = false "
            f"and (mimeType = '{FOLDER}' or mimeType = '{SHORTCUT}')"
        )
        found = list(_pages(service, query, "id, mimeType, shortcutDetails"))
        if not found:
            raise FileNotFoundError(
                f"the folder {'/'.join(parts)} was not found in My Drive of this Google "
                "account. Sign in with the account used on Colab, or add a shortcut to the "
                "shared folder under My Drive (Shared with me > right-click > Add shortcut)."
            )
        entry = found[0]
        parent = (
            entry["shortcutDetails"]["targetId"] if entry["mimeType"] == SHORTCUT else entry["id"]
        )
    return parent


@dataclass
class DriveMirror:
    """A local folder kept in step with a folder of the user's Google Drive."""

    service: object
    local_dir: Path
    drive_folder: tuple[str, ...] = ("cloudband_treino",)
    mirrored: tuple[str, ...] = MIRRORED
    settle_seconds: float = 60.0
    clock: Callable[[], float] = time.time
    log: Callable[[str], None] = print
    _folders: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _remote: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _pushed: dict[str, tuple[int, int]] = field(default_factory=dict, init=False, repr=False)
    _restored: bool = field(default=False, init=False, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _stop: threading.Event | None = field(default=None, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.local_dir = Path(self.local_dir)

    @property
    def where(self) -> str:
        return "My Drive/" + "/".join(self.drive_folder)

    # ---- the Drive API calls, kept thin so tests can replace them ----

    def _children(self, folder_id: str) -> list[dict]:
        query = f"'{folder_id}' in parents and trashed = false"
        fields = "id, name, mimeType, size, modifiedTime, shortcutDetails"
        return list(_pages(self.service, query, fields))

    def _create_folder(self, name: str, parent_id: str) -> str:
        body = {"name": name, "mimeType": FOLDER, "parents": [parent_id]}
        return self.service.files().create(body=body, fields="id").execute()["id"]

    def _download(self, file_id: str, dest: Path) -> None:
        from googleapiclient.http import MediaIoBaseDownload

        dest.parent.mkdir(parents=True, exist_ok=True)
        partial = dest.with_name(dest.name + ".partial")
        with open(partial, "wb") as handle:
            request = self.service.files().get_media(fileId=file_id)
            downloader = MediaIoBaseDownload(handle, request, chunksize=64 * 1024 * 1024)
            done = False
            while not done:
                _, done = downloader.next_chunk(num_retries=5)
        partial.replace(dest)

    def _upload(self, path: Path, name: str, parent_id: str, file_id: str | None) -> str:
        from googleapiclient.http import MediaFileUpload

        media = MediaFileUpload(
            str(path), mimetype="application/octet-stream", resumable=True,
            chunksize=64 * 1024 * 1024,
        )
        files = self.service.files()
        if file_id is not None:
            request = files.update(fileId=file_id, media_body=media, fields="id")
        else:
            body = {"name": name, "parents": [parent_id]}
            request = files.create(body=body, media_body=media, fields="id")
        reply = None
        while reply is None:
            _, reply = request.next_chunk(num_retries=5)
        return reply["id"]

    def _trash(self, file_id: str) -> None:
        self.service.files().update(fileId=file_id, body={"trashed": True}).execute()

    # ---- the mirror ----

    def _folder_id(self, relative_dir: str) -> str:
        """The id of a folder of the mirror, created on the Drive when missing."""
        if relative_dir in self._folders:
            return self._folders[relative_dir]
        parent = PurePosixPath(relative_dir).parent.as_posix()
        parent_id = self._folders[""] if parent == "." else self._folder_id(parent)
        name = PurePosixPath(relative_dir).name
        for child in self._children(parent_id):
            if child["name"] == name and child["mimeType"] in (FOLDER, SHORTCUT):
                found = (
                    child["shortcutDetails"]["targetId"]
                    if child["mimeType"] == SHORTCUT
                    else child["id"]
                )
                break
        else:
            found = self._create_folder(name, parent_id)
        self._folders[relative_dir] = found
        return found

    def _walk(self, relative_dir: str, folder_id: str, found: dict[str, list[dict]]) -> None:
        for child in self._children(folder_id):
            relative = f"{relative_dir}/{child['name']}"
            if child["mimeType"] in (FOLDER, SHORTCUT):
                target = (
                    child["shortcutDetails"]["targetId"]
                    if child["mimeType"] == SHORTCUT
                    else child["id"]
                )
                if child["mimeType"] == SHORTCUT and child["shortcutDetails"].get(
                    "targetMimeType"
                ) != FOLDER:
                    continue
                self._folders.setdefault(relative, target)
                self._walk(relative, target, found)
            elif not child["mimeType"].startswith("application/vnd.google-apps."):
                if not _skipped(child["name"]):
                    found.setdefault(relative, []).append(child)

    def restore(self) -> int:
        """Download results/ and checkpoints/ of the Drive folder. Returns the file count."""
        with self._lock:
            self.local_dir.mkdir(parents=True, exist_ok=True)
            self._folders = {"": resolve_path(self.service, self.drive_folder)}
            found: dict[str, list[dict]] = {}
            for top in self.mirrored:
                self._walk(top, self._folder_id(top), found)
            total = sum(int(pick_copy(c).get("size", 0)) for c in found.values())
            self.log(f"downloading {len(found)} file(s), {total / 1e9:.2f} GB, from {self.where}")
            self._remote, self._pushed = {}, {}
            for relative, copies in sorted(found.items()):
                chosen = pick_copy(copies)
                dest = self.local_dir / relative
                self._download(chosen["id"], dest)
                self._remote[relative] = chosen["id"]
                self._pushed[relative] = _signature(dest)
            for top in self.mirrored:
                (self.local_dir / top).mkdir(parents=True, exist_ok=True)
            self._restored = True
            return len(found)

    def push(self, settle_seconds: float | None = None) -> PushReport:
        """Send what changed since the last push or the restore."""
        if not self._restored:
            raise RuntimeError("restore() must succeed before the first push")
        settle = self.settle_seconds if settle_seconds is None else settle_seconds
        with self._lock:
            now = self.clock()
            current = local_files(self.local_dir, self.mirrored)
            uploaded, waiting = [], []
            for relative, path in current.items():
                try:
                    signature = _signature(path)
                except FileNotFoundError:
                    continue
                if self._pushed.get(relative) == signature:
                    continue
                if now - signature[1] / 1e9 < settle:
                    waiting.append(relative)
                    continue
                pure = PurePosixPath(relative)
                parent_id = self._folder_id(pure.parent.as_posix())
                self._remote[relative] = self._upload(
                    path, pure.name, parent_id, self._remote.get(relative)
                )
                self._pushed[relative] = signature
                uploaded.append(relative)
            deleted = []
            for relative in [r for r in self._pushed if r not in current]:
                file_id = self._remote.pop(relative, None)
                if file_id is not None:
                    self._trash(file_id)
                self._pushed.pop(relative)
                deleted.append(relative)
            return PushReport(tuple(uploaded), tuple(deleted), tuple(waiting))

    def push_all(self) -> PushReport:
        """Push without waiting for recent files to settle, for the end of a cell."""
        return self.push(settle_seconds=0.0)

    def start_background(self, interval_seconds: float = 300.0) -> None:
        """Push every interval from a daemon thread until stop_background()."""
        self.stop_background()
        stop = threading.Event()

        def loop() -> None:
            while not stop.wait(interval_seconds):
                try:
                    report = self.push()
                except Exception as error:  # retried at the next interval
                    self.log(f"[drive sync] push failed, retrying later: {error}")
                    continue
                if report.changed:
                    self.log(
                        f"[drive sync] {len(report.uploaded)} file(s) sent to the Drive, "
                        f"{len(report.deleted)} sent to its trash"
                    )

        self._stop = stop
        self._thread = threading.Thread(target=loop, name="drive-sync", daemon=True)
        self._thread.start()

    def stop_background(self) -> None:
        if self._stop is not None:
            self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._stop, self._thread = None, None


# ---- signing in with the user's own Google account ----

REDIRECT_URI = "http://localhost"


def client_config_from(text: str) -> dict:
    """The OAuth client of a client_secret*.json file downloaded from Google Cloud."""
    import json

    config = json.loads(text)
    if "installed" not in config:
        raise ValueError(
            "this is not the JSON of an OAuth client of type 'Desktop app' "
            "(it has no 'installed' section)"
        )
    return config


def start_sign_in(client_config: dict):
    """The flow and the URL the user opens to allow access to their Drive."""
    from google_auth_oauthlib.flow import Flow

    flow = Flow.from_client_config(
        client_config, scopes=[DRIVE_SCOPE], redirect_uri=REDIRECT_URI,
        autogenerate_code_verifier=True,
    )
    url, _ = flow.authorization_url(access_type="offline", prompt="consent")
    return flow, url


def code_from(pasted: str) -> str:
    """The code from the address the browser was sent to (or the bare code)."""
    from urllib.parse import parse_qs, urlparse

    pasted = pasted.strip()
    if "code=" not in pasted:
        return pasted
    values = parse_qs(urlparse(pasted).query).get("code")
    if not values:
        raise ValueError("no code= in the pasted address")
    return values[0]


def finish_sign_in(flow, pasted: str):
    flow.fetch_token(code=code_from(pasted))
    return flow.credentials


def load_credentials(token_json: str):
    """Saved credentials, refreshed when expired. None when they no longer work."""
    import json

    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    try:
        credentials = Credentials.from_authorized_user_info(json.loads(token_json), [DRIVE_SCOPE])
        if not credentials.valid:
            credentials.refresh(Request())
    except Exception:
        return None
    return credentials


def drive_service(credentials):
    from googleapiclient.discovery import build

    return build("drive", "v3", credentials=credentials, cache_discovery=False)


COLAB_MOUNT = PurePosixPath("/content/drive/MyDrive")


def tidy_directories(drive_folder: tuple[str, ...], subfolders: tuple[str, ...]):
    """The folders as cloudband.colab.drive_tidy names them: paths on the Colab mount."""
    return tuple(COLAB_MOUNT.joinpath(*drive_folder, *sub.split("/")) for sub in subfolders)
