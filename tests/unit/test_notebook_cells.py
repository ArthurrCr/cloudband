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


def test_the_l2a_test_is_secondary_and_kept_out_of_the_primary_comparison():
    path = next(p for p in NOTEBOOKS if p.name == "train_phase2_comparison.ipynb")
    cells = dict(code_cells(path))
    primary = next(s for s in cells.values() if 'evaluation.report(counts["test"])' in s)
    secondary = next(s for s in cells.values() if 'evaluation.report(counts["test_l2a"])' in s)
    scoring = next(s for s in cells.values() if "evaluation.score_models(" in s)

    assert "l2a" not in primary.lower()
    assert "never pooled" in secondary
    assert "-counts-{split}-seed{seed}" in scoring   # each level has its own cached file


def test_the_hugging_face_token_comes_right_after_the_drive_and_before_any_data():
    path = next(p for p in NOTEBOOKS if p.name == "train_phase2_comparison.ipynb")
    sources = [source for _, source in code_cells(path)]
    drive = next(i for i, s in enumerate(sources) if "drive.mount(" in s)
    token = next(i for i, s in enumerate(sources) if "whoami-v2" in s)
    first_read = next(i for i, s in enumerate(sources) if "dataset.load_remote(" in s)

    assert token == drive + 1
    assert token < first_read
