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

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from cloudband.colab.drive_tidy import COMPLETE_SHARE, FOLDER, SHORTCUT, _pages, _quote
from cloudband.train.store import RunStore

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
    """A local folder kept in step, both ways, with a folder of the user's Google Drive."""

    service: object
    local_dir: Path
    drive_folder: tuple[str, ...] = ("cloudband_treino",)
    mirrored: tuple[str, ...] = MIRRORED
    settle_seconds: float = 60.0
    clock: Callable[[], float] = time.time
    log: Callable[[str], None] = print
    _folders: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    # the newest Drive copy of every name, as last listed
    _seen: dict[str, dict] = field(default_factory=dict, init=False, repr=False)
    # the Drive version (id, modifiedTime) the local file was last made equal to
    _synced: dict[str, tuple[str, str]] = field(default_factory=dict, init=False, repr=False)
    # the local file (size, mtime) at that moment
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

    def _upload(self, path: Path, name: str, parent_id: str, file_id: str | None) -> dict:
        """Send a file; returns its id, modifiedTime and size on the Drive."""
        from googleapiclient.http import MediaFileUpload

        media = MediaFileUpload(
            str(path), mimetype="application/octet-stream", resumable=True,
            chunksize=64 * 1024 * 1024,
        )
        files = self.service.files()
        fields = "id, modifiedTime, size"
        if file_id is not None:
            request = files.update(fileId=file_id, media_body=media, fields=fields)
        else:
            body = {"name": name, "parents": [parent_id]}
            request = files.create(body=body, media_body=media, fields=fields)
        reply = None
        while reply is None:
            _, reply = request.next_chunk(num_retries=5)
        return reply

    def _trash(self, file_id: str) -> None:
        self.service.files().update(fileId=file_id, body={"trashed": True}).execute()

    # ---- listing the Drive ----

    @staticmethod
    def _target(child: dict) -> str:
        return child["shortcutDetails"]["targetId"] if child["mimeType"] == SHORTCUT else child["id"]

    @staticmethod
    def _is_folder(child: dict) -> bool:
        if child["mimeType"] == FOLDER:
            return True
        return child["mimeType"] == SHORTCUT and (
            child.get("shortcutDetails", {}).get("targetMimeType") == FOLDER
        )

    def _folder_id(self, relative_dir: str) -> str:
        """The id of a folder of the mirror, created on the Drive when missing."""
        if relative_dir in self._folders:
            return self._folders[relative_dir]
        parent = PurePosixPath(relative_dir).parent.as_posix()
        parent_id = self._folders[""] if parent == "." else self._folder_id(parent)
        name = PurePosixPath(relative_dir).name
        for child in self._children(parent_id):
            if child["name"] == name and self._is_folder(child):
                found = self._target(child)
                break
        else:
            found = self._create_folder(name, parent_id)
        self._folders[relative_dir] = found
        return found

    def _list(self, relative_dir: str, recursive: bool, found: dict[str, list[dict]]) -> None:
        for child in self._children(self._folder_id(relative_dir)):
            relative = f"{relative_dir}/{child['name']}"
            if self._is_folder(child):
                self._folders.setdefault(relative, self._target(child))
                if recursive:
                    self._list(relative, True, found)
            elif child["mimeType"].startswith("application/vnd.google-apps."):
                continue
            elif not _skipped(child["name"]):
                found.setdefault(relative, []).append(child)

    # ---- the mirror ----

    def _local_signature(self, relative: str) -> tuple[int, int] | None:
        try:
            return _signature(self.local_dir / relative)
        except FileNotFoundError:
            return None

    def _locally_changed(self, relative: str) -> bool:
        return self._local_signature(relative) != self._pushed.get(relative)

    def _pull_listed(self, listed: dict[str, dict], scope: Callable[[str], bool]) -> list[str]:
        """Bring down what changed on the Drive. A file changed here and not yet sent wins."""
        changed = []
        for relative, copy in sorted(listed.items()):
            self._seen[relative] = copy
            version = (copy["id"], copy["modifiedTime"])
            if self._synced.get(relative) == version:
                continue
            if relative in self._pushed and self._locally_changed(relative):
                continue
            if relative not in self._pushed and (self.local_dir / relative).exists():
                continue                     # made here and never sent: it will be sent
            dest = self.local_dir / relative
            self._download(copy["id"], dest)
            self._synced[relative], self._pushed[relative] = version, _signature(dest)
            changed.append(relative)
        for relative in [r for r in self._synced if scope(r) and r not in listed]:
            # gone from the Drive (a lock released, a run discarded by another session)
            self._seen.pop(relative, None)
            self._synced.pop(relative)
            if not self._locally_changed(relative):
                (self.local_dir / relative).unlink(missing_ok=True)
                self._pushed.pop(relative, None)
                changed.append(relative)
            else:
                self._pushed.pop(relative, None)   # sent again as a new file
        return changed

    def restore(self) -> int:
        """Download results/ and checkpoints/ of the Drive folder. Returns the file count."""
        with self._lock:
            self.local_dir.mkdir(parents=True, exist_ok=True)
            self._folders = {"": resolve_path(self.service, self.drive_folder)}
            found: dict[str, list[dict]] = {}
            for top in self.mirrored:
                self._list(top, True, found)
            listed = {relative: pick_copy(copies) for relative, copies in found.items()}
            total = sum(int(c.get("size", 0)) for c in listed.values())
            self.log(f"downloading {len(listed)} file(s), {total / 1e9:.2f} GB, from {self.where}")
            self._pull_listed(listed, scope=lambda _r: True)
            for top in self.mirrored:
                (self.local_dir / top).mkdir(parents=True, exist_ok=True)
            self._restored = True
            return len(listed)

    def pull(self, relative_paths: tuple[str, ...] | None = None) -> list[str]:
        """Bring down what other sessions changed: everything, or only the given files."""
        if not self._restored:
            raise RuntimeError("restore() must succeed before a pull")
        with self._lock:
            found: dict[str, list[dict]] = {}
            if relative_paths is None:
                for top in self.mirrored:
                    self._list(top, True, found)
                wanted = None
            else:
                wanted = set(relative_paths)
                for folder in sorted({PurePosixPath(r).parent.as_posix() for r in wanted}):
                    self._list(folder, False, found)
                found = {r: c for r, c in found.items() if r in wanted}
            listed = {relative: pick_copy(copies) for relative, copies in found.items()}
            return self._pull_listed(
                listed, scope=lambda r: wanted is None or r in wanted
            )

    def push(
        self, settle_seconds: float | None = None, relative_paths: tuple[str, ...] | None = None
    ) -> PushReport:
        """Send what changed here since the last sync: everything, or only the given files."""
        if not self._restored:
            raise RuntimeError("restore() must succeed before the first push")
        settle = self.settle_seconds if settle_seconds is None else settle_seconds
        with self._lock:
            now = self.clock()
            current = local_files(self.local_dir, self.mirrored)
            if relative_paths is not None:
                current = {r: p for r, p in current.items() if r in relative_paths}
            uploaded, waiting = [], []
            for relative in current:
                signature = self._local_signature(relative)
                if signature is None or self._pushed.get(relative) == signature:
                    continue
                if now - signature[1] / 1e9 < settle:
                    waiting.append(relative)
                    continue
                pure = PurePosixPath(relative)
                parent_id = self._folder_id(pure.parent.as_posix())
                known = self._seen.get(relative)
                reply = self._upload(
                    self.local_dir / relative, pure.name, parent_id, known and known["id"]
                )
                self._seen[relative] = reply
                self._synced[relative] = (reply["id"], reply["modifiedTime"])
                self._pushed[relative] = signature
                uploaded.append(relative)
            deleted = []
            gone = [
                r for r in self._pushed
                if r not in current and (relative_paths is None or r in relative_paths)
                and self._local_signature(r) is None
            ]
            for relative in gone:
                known = self._seen.pop(relative, None)
                if known is not None:
                    self._trash(known["id"])
                self._synced.pop(relative, None)
                self._pushed.pop(relative)
                deleted.append(relative)
            return PushReport(tuple(uploaded), tuple(deleted), tuple(waiting))

    def push_all(self) -> PushReport:
        """Push without waiting for recent files to settle, for the end of a cell."""
        return self.push(settle_seconds=0.0)

    def relative(self, path: Path) -> str:
        return Path(path).resolve().relative_to(self.local_dir.resolve()).as_posix()

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


@dataclass(frozen=True)
class DriveRunStore(RunStore):
    """RunStore on a local mirror of the Drive folder, safe with other sessions.

    Sessions (Colab or molab) take runs with lock files, as on the mounted Drive. Here
    every lock is read from the Drive and sent to it at once, the whole folder is
    pulled before a run is taken (so a run another session finished is seen as done),
    and the shared records (search results, curves, settings) are pulled right before
    they are rewritten, so the records of other sessions are kept.
    """

    mirror: DriveMirror | None = None

    def _rel(self, path: Path) -> tuple[str]:
        return (self.mirror.relative(path),)

    def wake(self) -> None:
        self.mirror.pull()

    def lock_info(self, name: str) -> dict | None:
        self.mirror.pull(self._rel(self.lock_path(name)))
        return super().lock_info(name)

    def _write_lock(self, name: str, started: float | None = None) -> float:
        now = super()._write_lock(name, started)
        self.mirror.push(settle_seconds=0.0, relative_paths=self._rel(self.lock_path(name)))
        return now

    def release(self, name: str) -> None:
        super().release(name)
        self.mirror.push(settle_seconds=0.0, relative_paths=self._rel(self.lock_path(name)))

    def _finished_elsewhere(self, name: str) -> bool:
        if self.is_done(name):
            return True
        for path in self.results_dir.glob("*-lr-search.partial.json"):
            try:
                records = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            for record in records:
                done = f"{record['run_id']}-seed{record['seed']}-lrsearch-{record['learning_rate']:g}"
                if done == name:
                    return True
        return False

    def claim(self, name: str, *args, **kwargs) -> bool:
        self.mirror.pull()
        if self._finished_elsewhere(name):
            return False             # done by another session since this one looked
        return super().claim(name, *args, **kwargs)

    def _shared_write(self, path: Path, write: Callable[[], None]) -> None:
        self.mirror.pull(self._rel(path))
        write()
        self.mirror.push(settle_seconds=0.0, relative_paths=self._rel(path))

    def save_search_run(self, protocol, run, budget, seed) -> None:
        self._shared_write(
            self.search_progress_path(protocol),
            lambda: super(DriveRunStore, self).save_search_run(protocol, run, budget, seed),
        )

    def save_curve(self, protocol, learning_rate, curve) -> None:
        self._shared_write(
            self.curves_path(protocol),
            lambda: super(DriveRunStore, self).save_curve(protocol, learning_rate, curve),
        )

    def save_setting(self, key, value) -> None:
        self._shared_write(
            self.settings_path(),
            lambda: super(DriveRunStore, self).save_setting(key, value),
        )

    def save_search(self, protocol, search):
        path = self.search_path(protocol)
        result = []
        self._shared_write(
            path, lambda: result.append(super(DriveRunStore, self).save_search(protocol, search))
        )
        return result[0]

    def save_run(self, run):
        paths = super().save_run(run)
        self.mirror.push(
            settle_seconds=0.0,
            relative_paths=tuple(self.mirror.relative(p) for p in paths),
        )
        return paths


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
