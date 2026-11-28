# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "fastai",
#     "timm",
#     "transformers",
#     "scipy",
#     "matplotlib",
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

    from cloudband.molab.drive_sync import (
        DriveMirror,
        client_config_from,
        drive_service,
        finish_sign_in,
        load_credentials,
        start_sign_in,
    )
    from cloudband.pipelines import l1c13_eval
    from cloudband.pipelines.phase2_eval import show
    from cloudband.train.store import RunStore

    pd.set_option("display.precision", 2)
    return (
        DriveMirror,
        RunStore,
        client_config_from,
        drive_service,
        finish_sign_in,
        l1c13_eval,
        load_credentials,
        pd,
        show,
        start,
        start_sign_in,
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
def _(DriveMirror, PROJECT_DIR, RunStore, WORK_DIR, drive, start):
    # The shared folder of the training, read through the Drive API: the counts of the
    # R-G-NIR models and of the published versions are read, the 13-band checkpoints
    # are scored, and what this notebook writes goes back in the last cell.
    DRIVE_FOLDER = ("cloudband_treino",)
    TRAINING_DIR = WORK_DIR / "cloudband_treino"
    RESULTS_DIR = TRAINING_DIR / "results"
    FIGURES_DIR = RESULTS_DIR / "figures"

    session = start(PROJECT_DIR, require_accelerator=True)
    drive_mirror = DriveMirror(drive, TRAINING_DIR, drive_folder=DRIVE_FOLDER)
    print(f"{drive_mirror.restore()} file(s) restored from {drive_mirror.where}")
    store = RunStore(
        results_dir=RESULTS_DIR,
        checkpoint_dir=TRAINING_DIR / "checkpoints",
        source_dir=PROJECT_DIR / "models",
    )
    return FIGURES_DIR, RESULTS_DIR, drive_mirror, session, store


@app.cell
def _(WORK_DIR, l1c13_eval):
    # All 13 bands of the 975 L1C test patches, about 7 GB, copied once to the session disk.
    import shutil

    test_table = l1c13_eval.load_test_table()
    _free = shutil.disk_usage(WORK_DIR).free
    print(f"L1C test: {len(test_table)} patches; {_free / 1e9:.0f} GB free on the disk")
    cache = l1c13_eval.build_test_cache(WORK_DIR / "cloudband_test_cache", test_table)
    return cache, test_table


@app.cell
def _(RESULTS_DIR, cache, l1c13_eval, store, test_table, torch):
    # Counts: the 13-band seeds and the RegNetZ seeds are scored here (or read back if
    # already saved); the R-G-NIR seeds and the published versions come from phase 2.
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    counts = l1c13_eval.score_trained(cache, test_table, store, RESULTS_DIR, DEVICE)
    rgn = l1c13_eval.load_rgn_counts(RESULTS_DIR, sorted(counts))
    published = l1c13_eval.load_published(RESULTS_DIR)
    regnetz = l1c13_eval.score_regnetz(cache, test_table, store, RESULTS_DIR, DEVICE)
    combined = l1c13_eval.combine(counts, rgn, published, regnetz)
    report = l1c13_eval.build_report(combined)
    _saved = l1c13_eval.save_report(report, RESULTS_DIR)
    print(f"seeds: {sorted(counts)}; {len(_saved)} tables saved to {RESULTS_DIR}")
    return combined, report


@app.cell
def _(l1c13_eval, report):
    # Median BOA per patch (%) of every model: the five seeds, their mean and SD.
    l1c13_eval.table_all_models(report)
    return


@app.cell
def _(l1c13_eval, report):
    # The comparison of the experiment: OCM (and each backbone) minus Swin, both at 13
    # bands, per ROI, in BOA points, with the 95% interval and the Holm-adjusted p-value.
    l1c13_eval.table_ocm_vs_swin(report)
    return


@app.cell
def _(l1c13_eval, report):
    # 13 bands minus R-G-NIR, per architecture, per ROI, in BOA points. Informative only:
    # the R-G-NIR models also trained on L2A, with the old crop (LIMITATIONS L2).
    l1c13_eval.table_bands(report)
    return


@app.cell
def _(l1c13_eval, report):
    # U-Net + RegNetZ-D8 (Focal Tversky) minus OCM and minus Swin, all at 13 bands, per
    # ROI. Whole systems: RegNetZ was trained with the earlier repository's recipe (D19).
    l1c13_eval.table_regnetz(report)
    return


@app.cell
def _(l1c13_eval, report):
    # OCM and Swin at 13 bands minus each published OmniCloudMask version, per ROI.
    l1c13_eval.table_vs_published(report)
    return


@app.cell
def _(FIGURES_DIR, l1c13_eval, report, show):
    # Where every model stands, per experiment: published versions, R-G-NIR, 13 bands.
    _figure = l1c13_eval.figure_all_models(report)
    show(_figure, FIGURES_DIR, l1c13_eval.figure_name("all-models"), lambda _f: None)
    _figure
    return


@app.cell
def _(combined, l1c13_eval):
    # Median BOA per patch (%) by share of the class in the patch, with the patch count.
    l1c13_eval.table_by_share(combined)
    return


@app.cell
def _(FIGURES_DIR, combined, l1c13_eval, show):
    # clear: BOA by share of clear pixels, published 4.0, each architecture at R-G-NIR
    # and at 13 bands, and RegNetZ.
    _figure = l1c13_eval.figure_by_share(combined, "clear")
    show(_figure, FIGURES_DIR, l1c13_eval.figure_name("by-share-clear"), lambda _f: None)
    _figure
    return


@app.cell
def _(FIGURES_DIR, combined, l1c13_eval, show):
    # cloud: BOA by share of cloud (thick + thin) pixels, same models.
    _figure = l1c13_eval.figure_by_share(combined, "cloud")
    show(_figure, FIGURES_DIR, l1c13_eval.figure_name("by-share-cloud"), lambda _f: None)
    _figure
    return


@app.cell
def _(FIGURES_DIR, combined, l1c13_eval, show):
    # shadow: BOA by share of cloud shadow pixels, same models.
    _figure = l1c13_eval.figure_by_share(combined, "shadow")
    show(_figure, FIGURES_DIR, l1c13_eval.figure_name("by-share-shadow"), lambda _f: None)
    _figure
    return


@app.cell
def _(combined, drive_mirror, report):
    # Sends the counts, tables and figures this notebook wrote to the shared folder.
    assert combined and report
    _report = drive_mirror.push_all()
    print(f"{len(_report.uploaded)} file(s) sent to the Drive")
    return


if __name__ == "__main__":
    app.run()
