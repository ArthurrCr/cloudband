"""The shared folder cloudband_treino in a Colab session, with or without the Drive mount.

The usual way is the Drive mount: My Drive appears at /content/drive and RunStore
writes straight into the shared folder. When the mount fails, as with Colab's
"credential propagation was unsuccessful", the session reaches the same folder through
the Drive API instead, signed in with the user's own Google account (OAuth, the same
sign-in molab uses): results/ and checkpoints/ are downloaded to a local folder at the
start, the notebook works on that folder, and push() sends back what it wrote.

The sign-in needs the client_secret JSON of an OAuth client of type "Desktop app"
(Google Cloud console), placed in /content through Colab's file panel. The token it
gives is saved in /content and reused until the runtime is deleted.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from cloudband.train.store import ensure_shared_folder

FOLDER_NAME = "cloudband_treino"
MOUNT_POINT = Path("/content/drive")
LOCAL_COPY = Path("/content") / FOLDER_NAME
TOKEN_FILE = Path("/content/gdrive_token.json")
CLIENT_FILES = Path("/content")


@dataclass
class SharedFolder:
    """Where the shared folder is in this session, and how to send results back."""

    root: Path
    mirror: object | None = None

    @property
    def mounted(self) -> bool:
        return self.mirror is None

    def push(self) -> list[str]:
        """Send to the Drive what the session wrote. Nothing to do on the mount."""
        if self.mirror is None:
            return []
        report = self.mirror.push(settle_seconds=0.0)
        return list(report.uploaded)


def _mount(mount_point: Path) -> None:
    from google.colab import drive

    drive.mount(str(mount_point))


def _credentials(log: Callable[[str], None], ask: Callable[[str], str]):
    from cloudband.molab.drive_sync import (
        client_config_from,
        finish_sign_in,
        load_credentials,
        start_sign_in,
    )

    if TOKEN_FILE.exists():
        credentials = load_credentials(TOKEN_FILE.read_text())
        if credentials is not None:
            log("Google: signed in with the token saved in this runtime")
            return credentials
    clients = sorted(CLIENT_FILES.glob("client_secret*.json"))
    if not clients:
        raise FileNotFoundError(
            "the Drive mount failed and there is no client_secret*.json in /content: "
            "upload the OAuth client JSON (Google Cloud, type Desktop app, the one used "
            "on molab) through the file panel on the left, then run this cell again"
        )
    flow, url = start_sign_in(client_config_from(clients[0].read_text()))
    log("1. Open this address, choose the account that sees cloudband_treino and allow:")
    log(url)
    log("2. The browser ends on a page that does not load (http://localhost...): copy its address.")
    credentials = finish_sign_in(flow, ask("3. Paste that address here: "))
    TOKEN_FILE.write_text(credentials.to_json())
    return credentials


def open_shared_folder(
    log: Callable[[str], None] = print,
    ask: Callable[[str], str] = input,
    mount: Callable[[Path], None] = _mount,
    mirror_factory: Callable[[], object] | None = None,
) -> SharedFolder:
    """The shared folder through the Drive mount, or through the Drive API if it fails."""
    try:
        mount(MOUNT_POINT)
        return SharedFolder(ensure_shared_folder(MOUNT_POINT / "MyDrive" / FOLDER_NAME))
    except Exception as error:
        log(f"Drive mount failed ({error}); using the Drive API instead")

    if mirror_factory is None:
        from cloudband.molab.drive_sync import DriveMirror, drive_service

        def mirror_factory():
            service = drive_service(_credentials(log, ask))
            return DriveMirror(service, LOCAL_COPY, drive_folder=(FOLDER_NAME,), log=log)

    mirror = mirror_factory()
    count = mirror.restore()
    log(f"{count} file(s) of results/ and checkpoints/ copied to {LOCAL_COPY}")
    return SharedFolder(Path(mirror.local_dir), mirror)
