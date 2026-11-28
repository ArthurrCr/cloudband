# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "fastai",
#     "timm",
#     "segmentation-models-pytorch",
#     "scipy",
#     "tacoreader",
#     "rasterio",
#     "pandas",
#     "requests",
#     "google-api-python-client",
#     "google-auth",
#     "google-auth-oauthlib",
# ]
# ///
import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    from pathlib import Path

    NOTEBOOK_DIR = Path(mo.notebook_dir() or Path.cwd()).resolve()
    return NOTEBOOK_DIR, Path, mo


@app.cell
def _(Path):
    import os
    import subprocess
    import sys

    REPO_URL = "https://github.com/ArthurrCr/cloudband.git"
    BRANCH = "main"
    # Everything big (code, cache, runs) goes under one folder of the session disk.
    WORK_DIR = Path(os.environ.get("CLOUDBAND_WORK_DIR", Path.home() / "cloudband_work"))
    PROJECT_DIR = WORK_DIR / "cloudband"
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    if not (PROJECT_DIR / ".git").exists():
        subprocess.run(
            ["git", "clone", "--quiet", "--branch", BRANCH, REPO_URL, str(PROJECT_DIR)], check=True
        )
    else:
        subprocess.run(["git", "-C", str(PROJECT_DIR), "fetch", "--quiet", "origin"], check=True)
        subprocess.run(
            ["git", "-C", str(PROJECT_DIR), "reset", "--quiet", "--hard", f"origin/{BRANCH}"],
            check=True,
        )

    SRC_DIR = str(PROJECT_DIR / "src")
    os.chdir(PROJECT_DIR)
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)
    return PROJECT_DIR, WORK_DIR, os


@app.cell
def _(os):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import pandas as pd
    import torch

    from cloudband.colab.session import reload_package, start

    reload_package("cloudband")

    from cloudband.datasets import cloudsen12 as dataset
    from cloudband.molab.drive_sync import (
        DriveMirror,
        DriveRunStore,
        client_config_from,
        drive_service,
        finish_sign_in,
        load_credentials,
        start_sign_in,
    )
    from cloudband.train.cache import LocalCache, measured_sample_bytes
    from cloudband.train.experiments import experiment, load_tables
    from cloudband.train.regnetz_recipe import RegNetZRecipe, loaders, train_all

    return (
        DriveMirror,
        DriveRunStore,
        LocalCache,
        RegNetZRecipe,
        client_config_from,
        dataset,
        drive_service,
        experiment,
        finish_sign_in,
        load_credentials,
        load_tables,
        loaders,
        measured_sample_bytes,
        pd,
        start,
        start_sign_in,
        torch,
        train_all,
    )


@app.cell
def _(NOTEBOOK_DIR, Path, load_credentials, mo, os):
    GOOGLE_TOKEN_FILE = Path(
        os.environ.get("GDRIVE_TOKEN_FILE", NOTEBOOK_DIR / "gdrive_token.json")
    )
    _saved = (
        GOOGLE_TOKEN_FILE.read_text()
        if GOOGLE_TOKEN_FILE.exists()
        else os.environ.get("GDRIVE_TOKEN", "")
    )
    saved_credentials = load_credentials(_saved) if _saved else None
    _clients = sorted(NOTEBOOK_DIR.glob("client_secret*.json"))
    client_text = (
        _clients[0].read_text() if _clients else os.environ.get("GDRIVE_CLIENT_SECRET_JSON", "")
    )
    client_file = mo.ui.file(filetypes=[".json"], label="client_secret JSON (Google Cloud, Desktop app)")
    if saved_credentials is not None:
        _shown = "Google: signed in with the saved token"
    elif client_text:
        _shown = "Google: OAuth client found, sign in below"
    else:
        _shown = client_file
    _shown
    return GOOGLE_TOKEN_FILE, client_file, client_text, saved_credentials


@app.cell
def _(
    client_config_from,
    client_file,
    client_text,
    mo,
    saved_credentials,
    start_sign_in,
):
    if saved_credentials is not None:
        sign_in_flow, pasted_box, _shown = None, None, None
    else:
        _text = client_text or (client_file.contents() or b"").decode()
        mo.stop(not _text, "Upload the client_secret JSON above to go on.")
        sign_in_flow, _url = start_sign_in(client_config_from(_text))
        pasted_box = mo.ui.text(
            label="After allowing, paste here the whole address of the page that did not load",
            full_width=True,
        )
        _shown = mo.vstack(
            [
                mo.Html(f'<a href="{_url}" target="_blank">1. Open this Google page and allow access to the Drive</a>'),
                pasted_box,
            ]
        )
    _shown
    return pasted_box, sign_in_flow


@app.cell
def _(
    GOOGLE_TOKEN_FILE,
    drive_service,
    finish_sign_in,
    mo,
    pasted_box,
    saved_credentials,
    sign_in_flow,
):
    if saved_credentials is not None:
        google_credentials = saved_credentials
    else:
        mo.stop(not pasted_box.value.strip(), "Waiting for the pasted address.")
        google_credentials = finish_sign_in(sign_in_flow, pasted_box.value)
        try:
            GOOGLE_TOKEN_FILE.write_text(google_credentials.to_json())
            print(f"token saved in {GOOGLE_TOKEN_FILE}; the next session skips the sign-in")
        except OSError as _error:
            print(f"token not saved ({_error}); the next session asks for the sign-in again")
    drive = drive_service(google_credentials)
    _me = drive.about().get(fields="user(emailAddress)").execute()["user"]["emailAddress"]
    print(f"Google Drive of {_me}")
    return drive, google_credentials


@app.cell
def _(mo):
    hf_token_box = mo.ui.text(kind="password", label="Hugging Face token, type Read (optional)", full_width=True)
    hf_token_box
    return (hf_token_box,)


@app.cell
def _(hf_token_box, os):
    import requests

    _token = hf_token_box.value.strip() or os.environ.get("HF_TOKEN", "").strip()
    os.environ.pop("HF_TOKEN", None)
    if _token:
        _reply = requests.get(
            "https://huggingface.co/api/whoami-v2",
            headers={"Authorization": f"Bearer {_token}"},
            timeout=30,
        )
        if _reply.status_code == 200:
            os.environ["HF_TOKEN"] = _token
            print(f"token accepted for the account {_reply.json().get('name', '?')}")
        else:
            print(f"token refused (HTTP {_reply.status_code}); going on without it")
    else:
        print("no token: requests use the shared anonymous quota")
    hf_checked = True
    return (hf_checked,)


@app.cell
def _(DriveMirror, DriveRunStore, PROJECT_DIR, WORK_DIR, drive, start):
    DRIVE_FOLDER = ("cloudband_treino",)
    TRAINING_DIR = WORK_DIR / "cloudband_treino"
    RESULTS_DIR = TRAINING_DIR / "results"
    CHECKPOINT_DIR = TRAINING_DIR / "checkpoints"

    session = start(PROJECT_DIR, require_accelerator=True)

    drive_mirror = DriveMirror(drive, TRAINING_DIR, drive_folder=DRIVE_FOLDER)
    print(f"{drive_mirror.restore()} file(s) restored from {drive_mirror.where}")
    drive_mirror.start_background(interval_seconds=300)

    store = DriveRunStore(
        results_dir=RESULTS_DIR,
        checkpoint_dir=CHECKPOINT_DIR,
        source_dir=PROJECT_DIR / "models",
        mirror=drive_mirror,
    )

    print(f"shared folder: {drive_mirror.where}  |  this session: {store.session_id}")
    return (
        DRIVE_FOLDER,
        RESULTS_DIR,
        TRAINING_DIR,
        drive_mirror,
        store,
    )


@app.cell
def _(dataset, experiment, hf_checked, load_tables):
    assert hf_checked

    # The same L1C tables and 13-band cache as the OCM and Swin experiment (l1c13).
    EXPERIMENT = experiment("l1c13")
    print(f"experiment {EXPERIMENT.name}: {EXPERIMENT.description}")

    for _level, _source in EXPERIMENT.remote_sources(dataset).items():
        _differences = dataset.verify_split_sizes(dataset.load_remote(_source))
        if _differences:
            print(f"{_level}: split sizes differ from documented ones:", _differences)

    tables = load_tables(EXPERIMENT, dataset, splits=("train", "validation", "test"))
    train_table = EXPERIMENT.tables(tables["train"])
    valid_table = EXPERIMENT.tables(tables["validation"])
    test_table = EXPERIMENT.tables(tables["test"])

    print("train patches:", len(train_table))
    print("validation patches:", len(valid_table))
    print("test patches (not read for training):", len(test_table))
    return EXPERIMENT, tables, test_table, train_table, valid_table


@app.cell
def _(
    EXPERIMENT,
    LocalCache,
    WORK_DIR,
    measured_sample_bytes,
    test_table,
    train_table,
    valid_table,
):
    import shutil

    from cloudband.datasets.fast_read import FastReader
    from cloudband.pipelines.phase2_eval import check_fast_reader

    CACHE_DIR = WORK_DIR / EXPERIMENT.cache_dir_name
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # molab gives 4 CPUs; the reads wait on the network, so more threads than cores still help
    WORKERS = 32
    # None decides by itself: compressed files only when the plain ones do not fit the disk
    COMPRESS = None

    fast = FastReader(bands=EXPERIMENT.reader_bands)
    _checked = check_fast_reader(
        fast,
        {"train": train_table, "validation": valid_table, "test": test_table},
        EXPERIMENT.reader_bands,
    )
    print(f"fast reader matches GDAL on {_checked} samples, {EXPERIMENT.band_count} bands")

    _probe = [fast(train_table, _i) for _i in (0, len(train_table) // 2, len(train_table) - 1)]
    assert all(_s.image.shape[0] == EXPERIMENT.band_count for _s in _probe)
    _count = len(train_table) + len(valid_table)
    _plain = measured_sample_bytes(_probe, bands_selected=True, compress=False)
    _small = measured_sample_bytes(_probe, bands_selected=True, compress=True)
    _free = shutil.disk_usage(CACHE_DIR).free
    _done = sum(1 for _ in CACHE_DIR.rglob("*.npz"))
    print(f"cache for {_count} patches: about {_count * _plain / 1e9:.0f} GB plain, "
          f"{_count * _small / 1e9:.0f} GB compressed; {_free / 1e9:.0f} GB free on the disk, "
          f"{_done} patches already copied")
    compress = COMPRESS if COMPRESS is not None else _free < 1.05 * (_count - _done) * _plain
    print("writing compressed files (slower to read)" if compress else "writing plain files")

    cache = LocalCache(CACHE_DIR)
    for _name, _table in (("train", train_table), ("validation", valid_table)):
        cache.build(
            _name,
            _table,
            read_sample=fast,
            workers=WORKERS,
            bands_selected=True,
            sample_bytes=_small if compress else _plain,
            compress=compress,
        )
    return (cache,)


@app.cell
def _(RegNetZRecipe, cache, loaders, store, torch, train_table, valid_table):
    # The recipe of github.com/ArthurrCr/Cloud-Segmentation (train/regnetz_recipe.py):
    # Focal Tversky, Adam 1e-4, ReduceLROnPlateau, early stopping, batch 4, no augmentation.
    recipe = RegNetZRecipe()
    NUM_WORKERS = 3

    train_reader = cache.reader("train", train_table)
    valid_reader = cache.reader("validation", valid_table)


    def data(seed):
        return loaders(
            train_table, valid_table, train_reader, valid_reader, recipe, seed,
            num_workers=NUM_WORKERS,
        )


    _images, _masks = next(iter(data(recipe.seeds[0])[0]))
    assert tuple(_images.shape[1:]) == (13, recipe.canvas, recipe.canvas), _images.shape
    print(f"batches {tuple(_images.shape)}, labels {tuple(_masks.shape)}, "
          f"image range {float(_images.min()):.3f} to {float(_images.max()):.3f}")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    for _seed in recipe.seeds:
        _name = recipe.checkpoint_name(_seed)
        _state = store.load_epoch_state(_name)
        if store.is_done(_name):
            _status = "done"
        elif _state is not None:
            _status = f"stopped after epoch {_state['epoch']}, resumes there"
        else:
            _status = "not started"
        print(f"{_name}: {_status}; lock: {store.describe_lock(_name)}")
    return DEVICE, data, recipe


@app.cell
def _(mo):
    train_button = mo.ui.run_button(label="Train the seeds that are missing", kind="success")
    train_button
    return (train_button,)


@app.cell
def _(DEVICE, data, drive_mirror, mo, recipe, store, train_all, train_button):
    mo.stop(not train_button.value, "Training waits for the button above.")
    try:
        trained = train_all(recipe, data, store, DEVICE)
    finally:
        _pushed = drive_mirror.push_all()
        print(f"sent to the Drive: {len(_pushed.uploaded)} file(s), "
              f"{len(_pushed.deleted)} to its trash")
    trained
    return


@app.cell
def _():
    import json

    return (json,)


@app.cell
def _(json, pd, recipe, store):
    # Each finished seed: epochs run, best epoch and its validation loss (Focal Tversky).
    _rows = []
    for _seed in recipe.seeds:
        _path = store.manifest_path(recipe.checkpoint_name(_seed))
        if _path.is_file():
            _run = json.loads(_path.read_text())
            _rows.append({
                "seed": _seed,
                "epochs run": _run["epochs_run"],
                "best epoch": _run["best_epoch"],
                "best validation loss": round(_run["best_val_loss"], 4),
                "stopped early": _run["stopped_early"],
            })
    pd.DataFrame(_rows)
    return


if __name__ == "__main__":
    app.run()
