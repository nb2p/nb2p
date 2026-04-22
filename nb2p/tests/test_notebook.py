from ..notebook import Notebook

a_code_strings = [
    "This is the first line\nThis is the second\nThe third",
    "Starting a new cell",
    "New cell again\nFinishing the cell",
    (
        "This is a cell with very looooooooooooooooooooooooooooooooooooooong"
        " lines\nFinish it!"
    ),
]


def test_notebook_from_code_strings():
    nb = Notebook.from_code_strings(a_code_strings)
    assert repr(nb) == "<Notebook(1 segments, 0 markdown cells, 4 code cells)>"

    assert (
        repr(nb.code_cells)
        == r"[<Cell(code, 3 lines, 'This is the first line\nThis is the second\nThe"
        r" third')>, "
        r"<Cell(code, 1 lines, 'Starting a new cell')>, "
        r"<Cell(code, 2 lines, 'New cell again\nFinishing the cell')>, "
        r"<Cell(code, 2 lines, 'This is a cell with very"
        r" looooooooooooooooooooooooooooooooooooooong lines\nFinish...')>]"
    )

    assert nb.code_strings == [
        "This is the first line\nThis is the second\nThe third",
        "Starting a new cell",
        "New cell again\nFinishing the cell",
        (
            "This is a cell with very looooooooooooooooooooooooooooooooooooooong"
            " lines\nFinish it!"
        ),
    ]
