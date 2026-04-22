from ..notebook import Notebook
from ..segment import CellEndSegmentsBuilder

a_code_strings = [
    "This is the first line\nThis is the second\nThe third",
    "Starting a new cell",
    "New cell again\nFinishing the cell",
    (
        "This is a cell with very looooooooooooooooooooooooooooooooooooooong"
        " lines\nFinish it!"
    ),
]


def test_segmentsbuilder_init():
    nb = Notebook.from_code_strings(a_code_strings)
    CellEndSegmentsBuilder(nb)
