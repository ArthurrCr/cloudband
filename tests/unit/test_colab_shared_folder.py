import types

import pytest

from cloudband.colab import shared_folder


def fake_mirror(local_dir, uploaded=("results/a.csv",)):
    calls = []

    def push(settle_seconds=None):
        calls.append(("push", settle_seconds))
        return types.SimpleNamespace(uploaded=uploaded)

    def restore():
        calls.append("restore")
        return 3

    return types.SimpleNamespace(local_dir=local_dir, push=push, restore=restore, calls=calls)


def test_the_mount_is_used_when_it_works(tmp_path, monkeypatch):
    (tmp_path / "MyDrive" / shared_folder.FOLDER_NAME).mkdir(parents=True)
    monkeypatch.setattr(shared_folder, "MOUNT_POINT", tmp_path)
    mounted = []

    shared = shared_folder.open_shared_folder(log=lambda m: None, mount=mounted.append)

    assert mounted == [tmp_path]
    assert shared.mounted
    assert shared.root == tmp_path / "MyDrive" / shared_folder.FOLDER_NAME
    assert shared.push() == []


def test_a_failed_mount_falls_back_to_a_restored_local_copy(tmp_path):
    def refuse(mount_point):
        raise RuntimeError("credential propagation was unsuccessful")

    mirror = fake_mirror(tmp_path / "copy")
    logged = []
    shared = shared_folder.open_shared_folder(
        log=logged.append, mount=refuse, mirror_factory=lambda: mirror
    )

    assert not shared.mounted
    assert shared.root == tmp_path / "copy"
    assert mirror.calls == ["restore"]           # downloaded before anything reads it
    assert shared.push() == ["results/a.csv"]
    assert mirror.calls[-1] == ("push", 0.0)     # nothing waits once the notebook is done
    assert any("credential propagation" in line for line in logged)


def test_without_a_client_file_the_fallback_says_what_to_upload(tmp_path, monkeypatch):
    monkeypatch.setattr(shared_folder, "TOKEN_FILE", tmp_path / "token.json")
    monkeypatch.setattr(shared_folder, "CLIENT_FILES", tmp_path)
    with pytest.raises(FileNotFoundError, match="client_secret"):
        shared_folder._credentials(log=lambda m: None, ask=lambda prompt: "")
