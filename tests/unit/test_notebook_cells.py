import ast
import json
from pathlib import Path

import pytest

NOTEBOOKS = sorted(Path(__file__).resolve().parents[2].glob("notebooks/**/*.ipynb"))


def code_cells(path):
    for index, cell in enumerate(json.loads(path.read_text())["cells"]):
        if cell["cell_type"] == "code":
            # shell escapes and magics are Colab syntax, not Python
            lines = [
                line[: len(line) - len(line.lstrip())] + "pass"
                if line.lstrip().startswith(("!", "%"))
                else line
                for line in "".join(cell["source"]).splitlines()
            ]
            yield index, "\n".join(lines)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_every_code_cell_is_valid_python(path):
    for index, source in code_cells(path):
        try:
            ast.parse(source)
        except SyntaxError as error:
            pytest.fail(f"{path.name}, cell {index}: {error}")


EVALUATION = "score_ocm_cloudsen12.ipynb"


def joined(name):
    path = next(p for p in NOTEBOOKS if p.name == name)
    return "\n".join(source for _, source in code_cells(path))


def test_the_l2a_test_is_secondary_and_kept_out_of_the_primary_comparison():
    path = next(p for p in NOTEBOOKS if p.name == EVALUATION)
    cells = dict(code_cells(path))
    primary = next(s for s in cells.values() if 'with_reference(counts["test"]' in s)
    secondary = next(s for s in cells.values() if 'with_reference(counts["test_l2a"]' in s)
    scoring = next(s for s in cells.values() if "evaluation.score_models(" in s)

    assert "l2a" not in primary.lower()
    assert "never pooled" in secondary
    assert "-counts-{split}-seed{seed}" in scoring   # each level has its own cached file


@pytest.mark.parametrize("name", ["train_phase2_comparison.ipynb", EVALUATION])
def test_the_hugging_face_token_comes_right_after_the_drive_and_before_any_data(name):
    path = next(p for p in NOTEBOOKS if p.name == name)
    sources = [source for _, source in code_cells(path)]
    drive = next(i for i, s in enumerate(sources) if "drive.mount(" in s)
    token = next(i for i, s in enumerate(sources) if "whoami-v2" in s)
    first_read = next(i for i, s in enumerate(sources) if "dataset.load_remote(" in s)

    assert token == drive + 1
    assert token < first_read


def test_the_phase2_evaluation_sits_beside_the_published_omnicloudmask():
    evaluate = joined(EVALUATION)
    assert "evaluation.score_models(" in evaluate
    assert "evaluation.score_predictor(" in evaluate and "ocm.predict_array(" in evaluate
    assert "published=PUBLISHED" in evaluate
    assert '("train"' not in evaluate and "run_everything" not in evaluate
    assert "evaluation.score_models(" not in joined("train_phase2_comparison.ipynb")


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebooks_leave_the_test_suite_to_the_repository(path):
    assert "pytest" not in "\n".join(source for _, source in code_cells(path))


def test_the_evaluation_draws_the_phase2_figures_after_the_tables():
    sources = [source for _, source in code_cells(
        next(p for p in NOTEBOOKS if p.name == EVALUATION)
    )]
    tables = next(i for i, s in enumerate(sources) if 'with_reference(counts["test_l2a"]' in s)
    figures = next(i for i, s in enumerate(sources) if "plot_paired_differences(" in s)
    assert figures > tables
    assert "plot_seed_boa(" in sources[figures] and "plot_unit_differences(" in sources[figures]
