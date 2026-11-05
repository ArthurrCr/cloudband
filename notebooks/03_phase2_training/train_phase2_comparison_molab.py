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
#     "huggingface-hub",
#     "pandas",
#     "requests",
# ]
# ///
import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Phase 2 training: OCM against Swin+UPerNet (molab)

    The same run as `train_phase2_comparison.ipynb`, written for [molab](https://molab.marimo.io).

    Before the first run:

    1. In the notebook specs button of the header, **turn the GPU on**.
    2. Have a Hugging Face token of type **Write**. molab has no Google Drive mount and its
       disk goes away with the session, so the shared folder of the Colab notebook becomes a
       **private dataset repository** on the Hugging Face Hub. The token reads CloudSEN12+ and
       keeps that repository.
    3. Training and maintenance only start from their buttons, so "run all" is safe.

    One session at a time per repository: two sessions pushing to the same repository would
    overwrite each other's runs.
    """)
    return


@app.cell
def _():
    import os
    import subprocess
    import sys
    from pathlib import Path

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
    from cloudband.eval.metrics import balanced_overall_accuracy
    from cloudband.models.ocm import PAPER_BACKBONES, OcmEnsemble, build_unet, ocm_search_protocol
    from cloudband.models.swin_upernet import build_swin_upernet
    from cloudband.molab.hub_sync import HubMirror
    from cloudband.pipelines.cloudsen12 import per_scene_metric, score_split
    from cloudband.train.cache import LocalCache
    from cloudband.train.comparison import compare_across_seeds
    from cloudband.train.data import build_dataloaders
    from cloudband.train.loop import checkpoint_name
    from cloudband.train.predictor import build_predictor
    from cloudband.train.protocol import ocm_shared_protocol, swin_shared_protocol
    from cloudband.train.schedule import phase2_networks, run_everything
    from cloudband.train.smoke import probe_micro_batch_size
    from cloudband.train.status import Plan, status_report
    from cloudband.train.store import RunStore, load_model_from_checkpoint

    return (
        HubMirror,
        LocalCache,
        OcmEnsemble,
        PAPER_BACKBONES,
        Plan,
        RunStore,
        balanced_overall_accuracy,
        build_dataloaders,
        build_predictor,
        build_swin_upernet,
        build_unet,
        checkpoint_name,
        compare_across_seeds,
        dataset,
        load_model_from_checkpoint,
        ocm_search_protocol,
        ocm_shared_protocol,
        pd,
        per_scene_metric,
        phase2_networks,
        probe_micro_batch_size,
        run_everything,
        score_split,
        start,
        status_report,
        swin_shared_protocol,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Hugging Face token

    Leave the box empty when the `HF_TOKEN` environment variable is already set. The repository
    name defaults to `<your account>/cloudband_treino`.
    """)
    return


@app.cell
def _(mo):
    token_box = mo.ui.text(kind="password", label="Hugging Face token (Write)", full_width=True)
    repo_box = mo.ui.text(label="Runs repository (blank: <account>/cloudband_treino)", full_width=True)
    mo.vstack([token_box, repo_box])
    return repo_box, token_box


@app.cell
def _(mo, os, repo_box, token_box):
    import requests

    _token = token_box.value.strip() or os.environ.get("HF_TOKEN", "").strip()
    mo.stop(not _token, mo.md("**Paste a Hugging Face token of type Write above to go on.**"))

    _reply = requests.get(
        "https://huggingface.co/api/whoami-v2",
        headers={"Authorization": f"Bearer {_token}"},
        timeout=30,
    )
    mo.stop(
        _reply.status_code != 200,
        mo.md(f"**Token refused (HTTP {_reply.status_code}).** Check it and paste it again."),
    )
    _account = _reply.json()
    _role = _account.get("auth", {}).get("accessToken", {}).get("role", "?")
    if _role == "read":
        print("warning: this is a Read token; the runs repository needs Write")
    os.environ["HF_TOKEN"] = _token
    HF_TOKEN = _token
    HUB_REPO = repo_box.value.strip() or f"{_account['name']}/cloudband_treino"
    print(f"token accepted for the account {_account.get('name', '?')} (role {_role})")
    print(f"runs repository: {HUB_REPO}")
    return HF_TOKEN, HUB_REPO


@app.cell
def _(HF_TOKEN, HUB_REPO, HubMirror, PROJECT_DIR, RunStore, WORK_DIR, start):
    TRAINING_DIR = WORK_DIR / "cloudband_treino"
    RESULTS_DIR = TRAINING_DIR / "results"
    CHECKPOINT_DIR = TRAINING_DIR / "checkpoints"

    session = start(PROJECT_DIR, require_accelerator=True)

    hub = HubMirror(HUB_REPO, TRAINING_DIR, token=HF_TOKEN)
    print(f"restored {hub.restore()} file(s) from {hub.url}")
    # every 10 minutes the new checkpoints and epoch states go up to the Hub, so a session
    # that ends (12 h limit, or 90 min idle) loses at most the epoch in progress
    hub.start_background(interval_seconds=600)

    store = RunStore(
        results_dir=RESULTS_DIR,
        checkpoint_dir=CHECKPOINT_DIR,
        source_dir=PROJECT_DIR / "models",
    )

    # True only when a run is reported "busy" and the session that held it is gone.
    BREAK_LOCKS = False

    print(f"local folder: {TRAINING_DIR}  |  this session: {store.session_id}")
    return BREAK_LOCKS, RESULTS_DIR, TRAINING_DIR, hub, store


@app.cell
def _(dataset, pd):
    def load_split(remote_source, split):
        table = dataset.load_remote(remote_source)
        differences = dataset.verify_split_sizes(table)
        if differences:
            print("split sizes differ from documented ones:", differences)
        return dataset.select(table, split=split)


    train_l1c = load_split(dataset.REMOTE_L1C, split="train")
    train_l2a = load_split(dataset.REMOTE_L2A, split="train")
    train_table = pd.concat([train_l1c, train_l2a], ignore_index=True)

    valid_l1c = load_split(dataset.REMOTE_L1C, split="validation")
    valid_l2a = load_split(dataset.REMOTE_L2A, split="validation")
    valid_table = pd.concat([valid_l1c, valid_l2a], ignore_index=True)

    test_table = load_split(dataset.REMOTE_L1C, split="test")
    test_l2a_table = load_split(dataset.REMOTE_L2A, split="test")

    print("train patches:", len(train_table))
    print("validation patches:", len(valid_table))
    print("test patches, L1C (primary):", len(test_table))
    print("test patches, L2A (secondary):", len(test_l2a_table))
    return (
        test_l2a_table,
        test_table,
        train_l1c,
        train_l2a,
        train_table,
        valid_l1c,
        valid_l2a,
        valid_table,
    )


@app.cell
def _(
    LocalCache,
    WORK_DIR,
    test_l2a_table,
    test_table,
    train_table,
    valid_table,
):
    import numpy as np

    from cloudband.datasets.cloudsen12 import read_sample as gdal_read_sample
    from cloudband.datasets.fast_read import FastReader
    from cloudband.pipelines.phase0 import select_rgn

    CACHE_DIR = WORK_DIR / "cloudband_cache"
    # molab gives 4 CPUs; the reads wait on the network, so more threads than cores still help
    WORKERS = 32

    fast = FastReader()

    _checks = (
        (train_table, 0),
        (train_table, len(train_table) - 1),
        (valid_table, 0),
        (test_table, 0),
        (test_l2a_table, 0),
    )
    for _table, _position in _checks:
        _expected, _got = gdal_read_sample(_table, _position), fast(_table, _position)
        _same = np.array_equal(_got.image, select_rgn(_expected.image)) and np.array_equal(
            _got.annotation, _expected.annotation
        )
        assert _same, f"the fast reader differs from GDAL at row {_position}"
    print(f"fast reader matches GDAL on {len(_checks)} samples")

    cache = LocalCache(CACHE_DIR)
    for _name, _table in (
        ("train", train_table),
        ("validation", valid_table),
        ("test", test_table),
        ("test_l2a", test_l2a_table),
    ):
        cache.build(_name, _table, read_sample=fast, workers=WORKERS, bands_selected=True)
    return cache, np


@app.cell
def _(
    cache,
    np,
    test_l2a_table,
    test_table,
    train_l1c,
    train_l2a,
    train_table,
    valid_l1c,
    valid_l2a,
):
    from cloudband.datasets.split_audit import audit_splits
    from cloudband.eval.level_similarity import compare_levels, level_pairs, summarize

    _audit = audit_splits(
        {
            ("L1C", "train"): train_l1c,
            ("L2A", "train"): train_l2a,
            ("L1C", "validation"): valid_l1c,
            ("L2A", "validation"): valid_l2a,
            ("L1C", "test"): test_table,
            ("L2A", "test"): test_l2a_table,
        }
    )
    print(_audit.summary)
    assert not _audit.leaked_rois, "a ROI appears in more than one split"

    LEVEL_PAIRS_SAMPLED = 200

    _pairs = level_pairs(train_l1c, train_l2a, l2a_offset=len(train_l1c))
    if not _pairs:
        print("\nno patch id is shared by the L1C and L2A tables; the level comparison is skipped")
        _shown = None
    else:
        _chosen = np.random.default_rng(0).choice(
            len(_pairs), size=min(LEVEL_PAIRS_SAMPLED, len(_pairs)), replace=False
        )
        _comparison = compare_levels(
            [_pairs[i] for i in sorted(_chosen)], train_table, cache.reader("train", train_table)
        )
        print(f"\nL1C against L2A of the same patch, {len(_chosen)} training patches")
        print("z_rmse: 0 = same image after normalization, 1.41 = unrelated")
        _shown = summarize(_comparison).round(3)
    _shown
    return


@app.cell
def _(
    PAPER_BACKBONES,
    build_dataloaders,
    build_swin_upernet,
    build_unet,
    cache,
    ocm_shared_protocol,
    probe_micro_batch_size,
    store,
    swin_shared_protocol,
    torch,
    train_table,
    valid_table,
):
    ocm_protocol = ocm_shared_protocol(learning_rate=1e-4)
    swin_protocol = swin_shared_protocol(learning_rate=1e-4)

    MICRO_BATCH_CANDIDATES = (8, 4, 2)
    NUM_WORKERS = 2

    _decided = store.load_setting("micro_batch")
    if _decided is None:
        ocm_micro_batch = min(
            probe_micro_batch_size(
                lambda name=_backbone: build_unet(name), ocm_protocol, MICRO_BATCH_CANDIDATES
            )
            for _backbone in PAPER_BACKBONES
        )
        swin_micro_batch = probe_micro_batch_size(
            build_swin_upernet, swin_protocol, MICRO_BATCH_CANDIDATES
        )
        store.save_setting(
            "micro_batch",
            {
                "ocm": ocm_micro_batch,
                "swin": swin_micro_batch,
                "decided_on": torch.cuda.get_device_name(0),
            },
        )
        print("micro-batch decided and saved:", store.load_setting("micro_batch"))
    else:
        ocm_micro_batch, swin_micro_batch = _decided["ocm"], _decided["swin"]
        for _backbone in PAPER_BACKBONES:
            probe_micro_batch_size(
                lambda name=_backbone: build_unet(name), ocm_protocol, (ocm_micro_batch,)
            )
        probe_micro_batch_size(build_swin_upernet, swin_protocol, (swin_micro_batch,))
        print(f"micro-batch from the runs repository (decided on {_decided['decided_on']}): "
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
    return ocm_dls, ocm_protocol, swin_dls, swin_protocol


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
            mo.md("**Learning-rate search**"),
            pd.DataFrame(_report.search).set_index(["model", "lr"])[_columns],
            mo.md("**Full runs**"),
            pd.DataFrame(_report.runs).set_index(["model", "seed"])[_columns + ["lr"]],
        ]
    )
    return (plans,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training

    Runs every learning-rate search and every seed that is not finished yet, resuming from the
    last saved epoch. The button keeps it from starting by accident (on "run all", or when a
    cell above runs again).
    """)
    return


@app.cell
def _(mo):
    train_button = mo.ui.run_button(label="Train everything that is missing", kind="success")
    train_button
    return (train_button,)


@app.cell
def _(
    BREAK_LOCKS,
    hub,
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
    mo.stop(not train_button.value, mo.md("Training waits for the button above."))

    _networks = phase2_networks(ocm_protocol, swin_protocol, ocm_dls, swin_dls)
    try:
        outcome = run_everything(
            _networks, store, seeds=ocm_protocol.seeds, break_locks=BREAK_LOCKS
        )
    finally:
        _pushed = hub.push_all("training stopped")
        print(f"pushed to {hub.url}: {len(_pushed.uploaded)} file(s), "
              f"{len(_pushed.deleted)} removed")
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
def _(
    OcmEnsemble,
    PAPER_BACKBONES,
    RESULTS_DIR,
    balanced_overall_accuracy,
    build_predictor,
    build_swin_upernet,
    build_unet,
    cache,
    checkpoint_name,
    load_model_from_checkpoint,
    ocm_protocol,
    pd,
    per_scene_metric,
    score_split,
    store,
    swin_protocol,
    test_table,
    torch,
):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


    def ocm_done(seed):
        return all(
            store.is_done(checkpoint_name(ocm_protocol, seed, backbone))
            for backbone in PAPER_BACKBONES
        )


    def swin_done(seed):
        return store.is_done(checkpoint_name(swin_protocol, seed))


    def load_ocm(seed):
        models = tuple(
            load_model_from_checkpoint(
                lambda backbone=backbone: build_unet(backbone, pretrained=False),
                store.checkpoint_path(checkpoint_name(ocm_protocol, seed, backbone)),
                DEVICE,
            )
            for backbone in PAPER_BACKBONES
        )
        return OcmEnsemble(models)


    def load_swin(seed):
        return load_model_from_checkpoint(
            lambda: build_swin_upernet(pretrained=False),
            store.checkpoint_path(checkpoint_name(swin_protocol, seed)),
            DEVICE,
        )


    def per_scene_boa(model, split="test", table=test_table):
        predictor = build_predictor(model, bands_selected=True)
        per_scene = score_split(cache.samples(split, table), predictor)
        return per_scene_metric(per_scene, balanced_overall_accuracy)


    def cached_boa(path, compute):
        if path.exists():
            return pd.read_csv(path, index_col=0)
        boa = compute()
        boa.to_csv(path)
        return boa


    seeds_ready = [s for s in ocm_protocol.seeds if ocm_done(s) and swin_done(s)]
    print("seeds finished for both architectures:", seeds_ready, "of", ocm_protocol.seeds)

    ocm_boa_per_seed, swin_boa_per_seed = [], []
    for _seed in seeds_ready:
        ocm_boa_per_seed.append(
            cached_boa(
                RESULTS_DIR / f"ocm_shared_per_scene_boa_seed{_seed}.csv",
                lambda seed=_seed: per_scene_boa(load_ocm(seed)),
            )
        )
        swin_boa_per_seed.append(
            cached_boa(
                RESULTS_DIR / f"swin_shared_per_scene_boa_seed{_seed}.csv",
                lambda seed=_seed: per_scene_boa(load_swin(seed)),
            )
        )

    for _arch, _frames in (("ocm", ocm_boa_per_seed), ("swin", swin_boa_per_seed)):
        print(f"{_arch} mean BOA per seed:")
        for _seed, _boa in zip(seeds_ready, _frames):
            print(f"  seed {_seed}:", _boa.mean().round(4).to_dict())
    return (
        cached_boa,
        load_ocm,
        load_swin,
        ocm_boa_per_seed,
        per_scene_boa,
        seeds_ready,
        swin_boa_per_seed,
    )


@app.cell
def _(compare_across_seeds, ocm_boa_per_seed, seeds_ready, swin_boa_per_seed):
    results = (
        compare_across_seeds(tuple(ocm_boa_per_seed), tuple(swin_boa_per_seed))
        if seeds_ready
        else []
    )
    if not seeds_ready:
        print("no seed is finished for both architectures yet")
    else:
        print(f"comparison over {len(seeds_ready)} seed(s): {seeds_ready}")
        for _result in results:
            print(_result.experiment)
            print("  pairable scenes:", _result.n_pairs)
            print("  p-value:", _result.p_value)
            print("  median difference, ocm minus swin:", _result.median_difference)
    return (results,)


@app.cell
def _(
    RESULTS_DIR,
    cached_boa,
    compare_across_seeds,
    hub,
    load_ocm,
    load_swin,
    per_scene_boa,
    results,
    seeds_ready,
    test_l2a_table,
):
    # Same scenes as the primary test at L2A: a robustness check, never pooled with it.
    if not seeds_ready:
        print("no seed is finished for both architectures yet")
    else:
        _ocm_boa_l2a, _swin_boa_l2a = [], []
        for _seed in seeds_ready:
            _ocm_boa_l2a.append(
                cached_boa(
                    RESULTS_DIR / f"ocm_shared_per_scene_boa_l2a_seed{_seed}.csv",
                    lambda seed=_seed: per_scene_boa(load_ocm(seed), "test_l2a", test_l2a_table),
                )
            )
            _swin_boa_l2a.append(
                cached_boa(
                    RESULTS_DIR / f"swin_shared_per_scene_boa_l2a_seed{_seed}.csv",
                    lambda seed=_seed: per_scene_boa(load_swin(seed), "test_l2a", test_l2a_table),
                )
            )

        results_l2a = compare_across_seeds(tuple(_ocm_boa_l2a), tuple(_swin_boa_l2a))
        print(f"L2A test (secondary), {len(seeds_ready)} seed(s): {seeds_ready}")
        for _primary, _secondary in zip(results, results_l2a):
            print(_secondary.experiment)
            print(f"  pairable scenes: L1C {_primary.n_pairs}, L2A {_secondary.n_pairs}")
            print(f"  p-value: L1C {_primary.p_value:.4g}, L2A {_secondary.p_value:.4g}")
            _same_side = (_primary.median_difference > 0) == (_secondary.median_difference > 0)
            print(
                "  median difference, ocm minus swin: "
                f"L1C {_primary.median_difference:+.4f}, L2A {_secondary.median_difference:+.4f}"
                f"  ({'same' if _same_side else 'DIFFERENT'} direction)"
            )
    _pushed = hub.push_all("per-scene results")
    print(f"\nresults pushed to {hub.url} ({len(_pushed.uploaded)} new file(s))")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Maintenance

    Off by default. `CONFIRM_DISCARD` deletes runs trained at another learning rate than their
    search chose; `CONFIRM_CLEANUP` deletes leftovers. Deleted files also leave the Hub
    repository on the next push (its history keeps them, so a mistake can be undone there).
    The Drive tidy of the Colab notebook has no counterpart here.
    """)
    return


@app.cell
def _(mo):
    CONFIRM_DISCARD = False
    CONFIRM_CLEANUP = False
    maintenance_button = mo.ui.run_button(label="Run maintenance")
    maintenance_button
    return CONFIRM_CLEANUP, CONFIRM_DISCARD, maintenance_button


@app.cell
def _(
    CONFIRM_CLEANUP,
    CONFIRM_DISCARD,
    TRAINING_DIR,
    hub,
    maintenance_button,
    mo,
    plans,
    status_report,
    store,
):
    mo.stop(not maintenance_button.value, mo.md("Maintenance waits for the button above."))

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

    _pushed = hub.push_all("maintenance")
    print(f"\nHub repository updated: {len(_pushed.uploaded)} sent, {len(_pushed.deleted)} removed")
    return


if __name__ == "__main__":
    app.run()
