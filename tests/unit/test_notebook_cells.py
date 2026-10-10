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
    primary = next(s for s in cells.values() if "results = compare_across_seeds" in s)
    secondary = next(s for s in cells.values() if "SECONDARY RESULT" in s)

    assert "l2a" not in primary.lower()
    assert "test_l2a" in secondary
    assert "_l2a_seed" in secondary          # its own cached files, not the primary ones
