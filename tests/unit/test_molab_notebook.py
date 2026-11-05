import ast
from pathlib import Path

NOTEBOOK = (
    Path(__file__).resolve().parents[2]
    / "notebooks/03_phase2_training/train_phase2_comparison_molab.py"
)


def source():
    return NOTEBOOK.read_text()


def test_the_molab_notebook_is_python_and_a_marimo_app():
    tree = ast.parse(source())
    assert any(
        isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "app"
        for node in tree.body
    )


def test_the_molab_notebook_has_no_colab_only_code_and_no_markdown_cells():
    text = source()
    assert "google.colab" not in text
    assert "drive.mount(" not in text
    assert "mo.md(" not in text


def test_training_and_maintenance_only_start_from_a_button():
    text = source()
    assert "mo.stop(not train_button.value" in text
    assert "mo.stop(not maintenance_button.value" in text


def test_it_uses_the_same_drive_folder_as_colab_and_pushes_after_training():
    text = source()
    assert 'DRIVE_FOLDER = ("cloudband_treino",)' in text
    assert text.index("drive_mirror.restore()") < text.index("store = RunStore(")
    training = text.index("run_everything(\n")
    assert text.index("drive_mirror.push_all()", training) > training
