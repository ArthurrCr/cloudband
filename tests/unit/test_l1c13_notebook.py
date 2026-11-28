import ast
import json
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



EVALUATION = NOTEBOOK.with_name("evaluate_l1c13_colab.ipynb")


def evaluation_code():
    cells = json.loads(EVALUATION.read_text())["cells"]
    return [
        "".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"
    ]


def test_the_evaluation_is_a_colab_notebook():
    metadata = json.loads(EVALUATION.read_text())["metadata"]
    assert metadata["accelerator"] == "GPU"
    code = "\n".join(evaluation_code())
    assert "open_shared_folder()" in code and "shared.push()" in code


def test_the_evaluation_notebook_only_calls_the_pipeline_in_order():
    cells = evaluation_code()
    for index, source in enumerate(cells):
        lines = [
            line[: len(line) - len(line.lstrip())] + "pass" if line.lstrip().startswith("!") else line
            for line in source.splitlines()
        ]
        tree = ast.parse("\n".join(lines))
        defined = [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        assert not defined, f"cell {index} defines {defined}; move it to src/cloudband"
    steps = [
        "l1c13_eval.load_test_table(", "l1c13_eval.build_test_cache(", "l1c13_eval.score_trained(",
        "l1c13_eval.load_rgn_counts(", "l1c13_eval.load_published(", "l1c13_eval.combine(",
        "l1c13_eval.save_report(", "figure_all_models(report)",
        'figure_by_share(combined, "clear")', 'figure_by_share(combined, "cloud")',
        'figure_by_share(combined, "shadow")', "table_ocm_vs_swin(", "table_bands(",
        "table_vs_published(", "shared.push()",
    ]
    positions = [next(i for i, s in enumerate(cells) if step in s) for step in steps]
    assert positions == sorted(positions)
    figures = [s for s in cells if "figure_" in s and "show(" in s]
    assert len(figures) == 4 and all(s.count("show(") == 1 for s in figures)


MOLAB_EVALUATION = NOTEBOOK.with_name("evaluate_l1c13_molab.py")


def test_the_molab_evaluation_only_calls_the_pipeline_and_sends_its_results():
    text = MOLAB_EVALUATION.read_text()
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


def test_the_evaluations_give_torch_a_device_it_accepts():
    """Session.device describes the GPU ("cuda:<name>"); torch.load refuses it as a device."""
    for text in (MOLAB_EVALUATION.read_text(), "\n".join(evaluation_code())):
        assert "session.device" not in text
        assert 'DEVICE = "cuda" if torch.cuda.is_available() else "cpu"' in text
