import ast
from pathlib import Path

NOTEBOOK = (
    Path(__file__).resolve().parents[2] / "notebooks/05_l1c13_training/train_l1c13_molab.py"
)


def source():
    return NOTEBOOK.read_text()


def test_the_notebook_is_a_marimo_app_without_colab_code():
    text = source()
    tree = ast.parse(text)
    assert any(
        isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "app"
        for node in tree.body
    )
    assert "google.colab" not in text
    assert "mo.md(" not in text


def test_it_trains_the_13_band_experiment_everywhere():
    text = source()
    assert 'EXPERIMENT = experiment("l1c13")' in text
    assert "FastReader(bands=EXPERIMENT.reader_bands)" in text
    assert "WORK_DIR / EXPERIMENT.cache_dir_name" in text
    assert "store.load_setting(EXPERIMENT.micro_batch_setting)" in text
    assert "experiment=EXPERIMENT.name" in text
    assert "ocm_shared_protocol" not in text and '"micro_batch"' not in text


def test_training_waits_for_the_button_and_pushes_to_the_drive():
    text = source()
    assert "mo.stop(not train_button.value" in text
    assert "mo.stop(not maintenance_button.value" in text
    assert 'DRIVE_FOLDER = ("cloudband_treino",)' in text
    training = text.index("run_everything(\n")
    assert text.index("drive_mirror.push_all()", training) > training


def test_the_test_split_is_only_audited_never_cached():
    text = source()
    assert '("train", train_table), ("validation", valid_table)' in text
    assert 'cache.build("test"' not in text


EVALUATION = NOTEBOOK.with_name("evaluate_l1c13_molab.py")


def test_the_evaluation_notebook_only_calls_the_pipeline_and_sends_its_results():
    text = EVALUATION.read_text()
    tree = ast.parse(text)
    assert any(
        isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "app"
        for node in tree.body
    )
    assert "google.colab" not in text and "mo.md(" not in text
    for call in (
        "l1c13_eval.build_test_cache(", "l1c13_eval.score_trained(", "l1c13_eval.load_rgn_counts(",
        "l1c13_eval.load_published(", "l1c13_eval.combine(", "l1c13_eval.save_report(",
    ):
        assert call in text
    for experiment in ("clear", "cloud", "shadow"):
        assert f'figure_by_share(combined, "{experiment}")' in text
    assert text.index("drive_mirror.push_all()") > text.index('figure_by_share(combined, "shadow")')
