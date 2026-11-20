# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "fastai",
#     "timm",
#     "transformers",
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
    from cloudband.models.ocm import PAPER_BACKBONES, build_unet, ocm_search_protocol
    from cloudband.models.swin_upernet import build_swin_upernet
    from cloudband.colab import drive_tidy
    from cloudband.molab.drive_sync import (
        DriveMirror,
        DriveRunStore,
        client_config_from,
        drive_service,
        finish_sign_in,
        load_credentials,
        start_sign_in,
        tidy_directories,
    )
    from cloudband.train.cache import LocalCache, measured_sample_bytes
    from cloudband.train.data import build_dataloaders
    from cloudband.train.experiments import experiment, load_tables
    from cloudband.train.loop import checkpoint_name
    from cloudband.train.schedule import phase2_networks, run_everything
    from cloudband.train.smoke import loader_samples_per_second, probe_micro_batch_size
    from cloudband.train.status import Plan, status_report

    return (
        DriveMirror,
        DriveRunStore,
        LocalCache,
        PAPER_BACKBONES,
        Plan,
        build_dataloaders,
        build_swin_upernet,
        build_unet,
        checkpoint_name,
        client_config_from,
        dataset,
        drive_service,
        drive_tidy,
        experiment,
        finish_sign_in,
        load_credentials,
        load_tables,
        loader_samples_per_second,
        measured_sample_bytes,
        ocm_search_protocol,
        pd,
        phase2_networks,
        probe_micro_batch_size,
        run_everything,
        start,
        start_sign_in,
        status_report,
        tidy_directories,
        torch,
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

    # True only when a run is reported "busy" and the session that held it is gone.
    BREAK_LOCKS = False

    print(f"shared folder: {drive_mirror.where}  |  this session: {store.session_id}")
    return (
        BREAK_LOCKS,
        DRIVE_FOLDER,
        RESULTS_DIR,
        TRAINING_DIR,
        drive_mirror,
        store,
    )


@app.cell
def _(dataset, experiment, hf_checked, load_tables):
    assert hf_checked

    # All 13 bands of L1C (B01 to B12 with B8A and B10), trained on L1C only: L2A has
    # no B10 (DIVERGENCES D18). Same protocol, grid and seeds as the R-G-NIR models.
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
def _(tables):
    from cloudband.datasets.split_audit import audit_splits

    _audit = audit_splits(
        {(level, split): tables[split][level] for split in tables for level in tables[split]}
    )
    print(_audit.summary)
    assert not _audit.leaked_rois, "a ROI appears in more than one split"
    return


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
def _(
    EXPERIMENT,
    PAPER_BACKBONES,
    build_dataloaders,
    build_swin_upernet,
    build_unet,
    cache,
    loader_samples_per_second,
    probe_micro_batch_size,
    store,
    torch,
    train_table,
    valid_table,
):
    ocm_protocol, swin_protocol = EXPERIMENT.protocols()
    BANDS = EXPERIMENT.band_count

    MICRO_BATCH_CANDIDATES = (8, 4, 2)
    # 13 bands cost about four times the CPU of 3 per sample to augment; molab has 4 cores
    NUM_WORKERS = 3


    def ocm_builder(name):
        return lambda: build_unet(name, in_chans=BANDS)


    def swin_builder():
        return build_swin_upernet(in_chans=BANDS)


    _decided = store.load_setting(EXPERIMENT.micro_batch_setting)
    if _decided is None:
        ocm_micro_batch = min(
            probe_micro_batch_size(ocm_builder(_backbone), ocm_protocol, MICRO_BATCH_CANDIDATES)
            for _backbone in PAPER_BACKBONES
        )
        swin_micro_batch = probe_micro_batch_size(
            swin_builder, swin_protocol, MICRO_BATCH_CANDIDATES
        )
        store.save_setting(
            EXPERIMENT.micro_batch_setting,
            {
                "ocm": ocm_micro_batch,
                "swin": swin_micro_batch,
                "decided_on": torch.cuda.get_device_name(0),
            },
        )
        print("micro-batch decided and saved:", store.load_setting(EXPERIMENT.micro_batch_setting))
    else:
        ocm_micro_batch, swin_micro_batch = _decided["ocm"], _decided["swin"]
        for _backbone in PAPER_BACKBONES:
            probe_micro_batch_size(ocm_builder(_backbone), ocm_protocol, (ocm_micro_batch,))
        probe_micro_batch_size(swin_builder, swin_protocol, (swin_micro_batch,))
        print(f"micro-batch from the shared folder (decided on {_decided['decided_on']}): "
              f"OCM {ocm_micro_batch}, Swin {swin_micro_batch}; it fits this GPU")

    train_reader = cache.reader("train", train_table)
    valid_reader = cache.reader("validation", valid_table)


    def make_dataloaders(micro_batch_size, protocol):
        return build_dataloaders(
            train_table=train_table,
            valid_table=valid_table,
            micro_batch_size=micro_batch_size,
            num_workers=NUM_WORKERS,
            read_sample=train_reader,
            valid_read_sample=valid_reader,
            bands_selected=True,
            augmentation=protocol.augmentation if protocol.augment else None,
        )


    ocm_dls = make_dataloaders(ocm_micro_batch, ocm_protocol)
    swin_dls = make_dataloaders(swin_micro_batch, swin_protocol)

    _batch = next(iter(ocm_dls.train))[0]
    assert _batch.shape[1] == BANDS, f"batches carry {_batch.shape[1]} bands, not {BANDS}"
    print(f"batches carry {BANDS} bands; the input pipeline alone delivers "
          f"{loader_samples_per_second(ocm_dls):.0f} training samples/s")
    return NUM_WORKERS, ocm_dls, ocm_protocol, swin_dls, swin_protocol


@app.cell
def _(
    PAPER_BACKBONES,
    Plan,
    mo,
    ocm_protocol,
    ocm_search_protocol,
    pd,
    status_report,
    store,
    swin_protocol,
):
    plans = [
        Plan(
            f"OCM {backbone}",
            ocm_protocol,
            backbone,
            search_protocol=ocm_search_protocol(ocm_protocol, backbone),
        )
        for backbone in PAPER_BACKBONES
    ]
    plans.append(Plan("Swin+UPerNet", swin_protocol))
    _report = status_report(store, plans)

    print(_report.summary)
    _columns = ["state", "epochs", "best_val_loss", "session", "note"]
    mo.vstack(
        [
            "**Learning-rate search**",
            pd.DataFrame(_report.search).set_index(["model", "lr"])[_columns],
            "**Full runs**",
            pd.DataFrame(_report.runs).set_index(["model", "seed"])[_columns + ["lr"]],
        ]
    )
    return (plans,)


@app.cell
def _(mo):
    from cloudband.molab import workers as parallel

    # each worker brings its own loader processes, and molab has 4 cores
    worker_count = mo.ui.slider(1, 4, value=1, label="parallel workers", show_value=True)
    parallel_button = mo.ui.run_button(label="Start parallel workers", kind="success")
    stop_workers_button = mo.ui.run_button(label="Stop workers", kind="danger")
    mo.hstack([worker_count, parallel_button, stop_workers_button], justify="start")
    return parallel, parallel_button, stop_workers_button, worker_count


@app.cell
def _(
    EXPERIMENT,
    NUM_WORKERS,
    PROJECT_DIR,
    TRAINING_DIR,
    WORK_DIR,
    drive_mirror,
    google_credentials,
    mo,
    parallel,
    parallel_button,
    store,
    worker_count,
):
    mo.stop(not parallel_button.value, "Parallel workers wait for the button above.")
    assert store.load_setting(EXPERIMENT.micro_batch_setting) is not None, (
        "run the micro-batch cell first"
    )

    WORKER_TOKEN_FILE = WORK_DIR / "gdrive_token.json"
    WORKER_TOKEN_FILE.write_text(google_credentials.to_json())
    drive_mirror.save_state()
    started_workers = parallel.launch(
        worker_count.value,
        WORK_DIR,
        PROJECT_DIR,
        TRAINING_DIR,
        WORKER_TOKEN_FILE,
        num_workers=NUM_WORKERS,
        experiment=EXPERIMENT.name,
    )
    print(f"{len(started_workers)} worker(s) started; logs in {WORK_DIR}/worker<k>.log")
    return (started_workers,)


@app.cell
def _(mo, started_workers, stop_workers_button):
    if stop_workers_button.value:
        for _worker in started_workers:
            _worker.stop()
        print("workers asked to stop; each sends its files to the Drive before exiting")
    worker_refresh = mo.ui.refresh(default_interval="30s", label="refresh")
    worker_refresh
    return (worker_refresh,)


@app.cell
def _(mo, parallel, started_workers, worker_refresh):
    worker_refresh
    mo.vstack(
        [mo.plain_text(parallel.gpu_summary() + "\n" + parallel.cpu_summary())]
        + [
            mo.vstack([mo.plain_text(f"worker {_w.index}: {_w.state}"), mo.plain_text(_w.tail())])
            for _w in started_workers
        ]
    )
    return


@app.cell
def _(mo):
    train_button = mo.ui.run_button(label="Train everything that is missing", kind="success")
    train_button
    return (train_button,)


@app.cell
def _(
    BREAK_LOCKS,
    drive_mirror,
    mo,
    ocm_dls,
    ocm_protocol,
    phase2_networks,
    run_everything,
    store,
    swin_dls,
    swin_protocol,
    train_button,
):
    mo.stop(not train_button.value, "Training waits for the button above.")

    _networks = phase2_networks(ocm_protocol, swin_protocol, ocm_dls, swin_dls)
    try:
        outcome = run_everything(
            _networks, store, seeds=ocm_protocol.seeds, break_locks=BREAK_LOCKS
        )
    finally:
        _pushed = drive_mirror.push_all()
        print(f"sent to the Drive: {len(_pushed.uploaded)} file(s), "
              f"{len(_pushed.deleted)} to its trash")
    outcome
    return


@app.cell
def _(
    PAPER_BACKBONES,
    checkpoint_name,
    ocm_protocol,
    pd,
    store,
    swin_protocol,
):
    _names = [
        checkpoint_name(ocm_protocol, seed, backbone)
        for seed in ocm_protocol.seeds
        for backbone in PAPER_BACKBONES
    ] + [checkpoint_name(swin_protocol, seed) for seed in swin_protocol.seeds]

    _convergence = pd.DataFrame(
        [summary for summary in (store.convergence(name) for name in _names) if summary]
    )
    if _convergence.empty:
        print("no finished run with a recorded loss history yet")
    else:
        _flagged = int(_convergence.still_improving.sum())
        print(f"{_flagged} of {len(_convergence)} runs were still improving at the end")
    _convergence
    return


@app.cell
def _(mo):
    # CONFIRM_DISCARD: delete runs trained at another learning rate than their search chose.
    # CONFIRM_CLEANUP: delete leftovers in the shared folder.
    # DRIVE_TIDY "off", "list" or "trash": stale Drive copies left by Colab.
    CONFIRM_DISCARD = False
    CONFIRM_CLEANUP = False
    DRIVE_TIDY = "off"
    ACCEPT_PARTIAL_CURVES = False
    maintenance_button = mo.ui.run_button(label="Run maintenance")
    maintenance_button
    return (
        ACCEPT_PARTIAL_CURVES,
        CONFIRM_CLEANUP,
        CONFIRM_DISCARD,
        DRIVE_TIDY,
        maintenance_button,
    )


@app.cell
def _(
    ACCEPT_PARTIAL_CURVES,
    CONFIRM_CLEANUP,
    CONFIRM_DISCARD,
    DRIVE_FOLDER,
    DRIVE_TIDY,
    TRAINING_DIR,
    drive,
    drive_mirror,
    drive_tidy,
    maintenance_button,
    mo,
    ocm_protocol,
    plans,
    status_report,
    store,
    swin_protocol,
    tidy_directories,
):
    mo.stop(not maintenance_button.value, "Maintenance waits for the button above.")

    _report = status_report(store, plans)
    _to_discard = [row for row in _report.runs if row["at_other_rate"]]
    if not _to_discard:
        print("nothing to discard: every run is at the rate its search chose")
    for _row in _to_discard:
        print(f"{_row['model']} seed {_row['seed']}: trained at {_row['lr']:g}, "
              f"chosen {_row['chosen_lr']:g} ({_row['state']}, {_row['epochs']})")
        if _row["state"] == "running":
            print("   skipped: a session is training it right now")
        elif CONFIRM_DISCARD:
            _removed = store.discard(_row["name"])
            print(f"   discarded ({len(_removed)} files)")
    if _to_discard and not CONFIRM_DISCARD:
        print("\nnothing was deleted: set CONFIRM_DISCARD = True and run again to discard")

    _unused = store.unused_files()


    def _size(path):
        return path.stat().st_size / 1e6


    for _path, _reason in _unused:
        print(f"{_size(_path):9.1f} MB  {_path.name}\n             {_reason}")
    print(f"\n{len(_unused)} file(s), {sum(_size(p) for p, _ in _unused):.1f} MB")
    if _unused and CONFIRM_CLEANUP:
        print(f"removed {store.remove_unused(_unused)} file(s)")
    elif _unused:
        print("nothing was deleted: set CONFIRM_CLEANUP = True and run again to remove them")

    _known = {path for path, _ in _unused}
    print("\nlargest files that stay:")
    for _path, _nbytes in store.largest_files(TRAINING_DIR, exclude=_known):
        print(f"{_nbytes / 1e6:9.1f} MB  {_path.relative_to(TRAINING_DIR)}")

    _pushed = drive_mirror.push_all()
    print(f"\nDrive updated: {len(_pushed.uploaded)} sent, {len(_pushed.deleted)} to its trash")

    drive_tidy.run_tidy(
        DRIVE_TIDY,
        store,
        plans,
        tidy_directories(DRIVE_FOLDER, ("results", "checkpoints", "results/progress")),
        (ocm_protocol.run_id, swin_protocol.run_id),
        service=drive,
        accept_partial=ACCEPT_PARTIAL_CURVES,
    )
    return


if __name__ == "__main__":
    app.run()
