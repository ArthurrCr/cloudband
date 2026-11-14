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


def test_the_evaluation_notebook_only_calls_the_project_code():
    path = next(p for p in NOTEBOOKS if p.name == EVALUATION)
    for index, source in code_cells(path):
        tree = ast.parse(source)
        defined = [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
        loops = [n for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While))]
        assert not defined, f"cell {index} defines {defined}; move it to src/cloudband"
        assert len(loops) <= 1, f"cell {index} has {len(loops)} loops; move the logic to src/cloudband"


def test_the_evaluation_runs_the_phase2_steps_in_order():
    sources = [s for _, s in code_cells(next(p for p in NOTEBOOKS if p.name == EVALUATION))]
    steps = [
        "load_test_tables(", "build_test_cache(", "phase2_eval.score_published(",
        "score_trained(", "build_reports(", "save_reports(", "figure_all_models(reports, \"L1C\")",
        "figure_all_models(reports, \"L2A\")", "figure_levels(",
    ]
    positions = [next(i for i, s in enumerate(sources) if step in s) for step in steps]
    assert positions == sorted(positions)


@pytest.mark.parametrize("name", ["train_phase2_comparison.ipynb", EVALUATION])
def test_the_hugging_face_token_comes_right_after_the_drive_and_before_any_data(name):
    path = next(p for p in NOTEBOOKS if p.name == name)
    sources = [source for _, source in code_cells(path)]
    drive = next(i for i, s in enumerate(sources) if "drive.mount(" in s)
    token = next(i for i, s in enumerate(sources) if "whoami-v2" in s)
    first_read = next(
        i for i, s in enumerate(sources) if "load_remote(" in s or "load_test_tables(" in s
    )

    assert token == drive + 1
    assert token < first_read


def test_the_phase2_evaluation_sits_beside_the_published_omnicloudmask():
    evaluate = joined(EVALUATION)
    assert "phase2_eval.score_trained(" in evaluate and "phase2_eval.score_published(" in evaluate
    assert '("train"' not in evaluate and "run_everything" not in evaluate
    assert "score_models(" not in joined("train_phase2_comparison.ipynb")


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebooks_leave_the_test_suite_to_the_repository(path):
    assert "pytest" not in "\n".join(source for _, source in code_cells(path))


def test_every_figure_of_the_evaluation_has_a_cell_of_its_own_under_a_heading():
    cells = json.loads(next(p for p in NOTEBOOKS if p.name == EVALUATION).read_text())["cells"]
    kinds = []
    for index, cell in enumerate(cells):
        source = "".join(cell["source"])
        if cell["cell_type"] == "markdown":
            kinds.append("heading")
        elif "phase2_eval.show(" in source:
            assert source.count("phase2_eval.show(") == 1, f"cell {index} shows several figures"
            kinds.append("figure")
    figures = [i for i, kind in enumerate(kinds) if kind == "figure"]
    assert len(figures) >= 11
    for before, after in zip([-1] + figures, figures):
        assert "heading" in kinds[before + 1:after], "two figures with no heading between them"
