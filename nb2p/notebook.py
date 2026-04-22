"""Computational notebook classes."""

import itertools
from typing import List, Literal, Optional

import pandas as pd
from typing_extensions import Self

from nb2p import astparse

from . import codeop
from .dataset import CellD

p, lang = astparse.parser()


def extract_notebook_ast_sequence(notebook: "Notebook", parser: astparse.Parser):
    result = []

    ast_tree = parser.parse("\n".join(notebook.code_lines).encode())
    cursor = ast_tree.root_node.walk()

    cursor.goto_first_child()

    gs = cursor.node
    num_ast_children = 1

    code_tokens = astparse.node_to_code_token(gs, notebook.code_lines)

    # print(f"> Node {num_ast_children}")
    # print(f"> Points: {gs.start_point} {gs.end_point}")
    # print(f"> Content: ")
    # print(code_tokens)
    # print("----------------------------------------")

    result.append({"end_line_number": gs.end_point[0], "code": code_tokens})

    while cursor.goto_next_sibling():
        gs = cursor.node
        num_ast_children += 1

        code_tokens = astparse.node_to_code_token(gs, notebook.code_lines)

        result.append({"end_line_number": gs.end_point[0], "code": code_tokens})

        # print(f"> Node {num_ast_children}")
        # print(f"> Points: {gs.start_point} {gs.end_point}")
        # print(f"> Content: ")
        # print(code_tokens)
        # print("----------------------------------------")

    return result


class Cell:
    REPR_LIMIT = 80

    def __init__(
        self,
        type: Literal["markdown", "code"],
        content: List[str],
    ) -> None:
        self._type = type
        self._content = content

    @property
    def type(self):
        return self._type

    @property
    def content(self) -> List[str]:
        """Get cell content in a list of strings."""
        return self._content

    @content.setter
    def content(self, value) -> None:
        self._content = value

    def __len__(self) -> int:
        return len(self._content)

    def __str__(self) -> str:
        """Get cell content in a newline joined string."""
        return "\n".join(self._content)

    def remove_comment(self) -> Self:
        if self._type == "code":
            result = codeop.remove_comments_and_docstrings("\n".join(self._content))
            result = result.split("\n")
            # self._content = [r for r in result if r.strip()]
            self._content = result
        return self

    def __repr__(self) -> str:
        """Print cell information and content preview."""
        content_preview = (
            f"{str(self)[: self.REPR_LIMIT]}..."
            if len(str(self)) > self.REPR_LIMIT
            else str(self)
        )
        return (
            f"<Cell({self._type}, {len(self._content)} lines, {repr(content_preview)})>"
        )


class Segment:
    def __init__(
        self, cells: List[Cell], has_code: bool, num_ast_children: Optional[int] = None
    ) -> None:
        self._cells = cells
        self._has_code = has_code
        if num_ast_children is not None:
            self._num_ast_children = num_ast_children
        else:
            code = "\n".join([str(cell) for cell in self._cells if cell.type == "code"])
            self._num_ast_children = codeop.get_num_ast_children(code, p)

    @property
    def cells(self):
        return self._cells

    @property
    def has_code(self):
        return self._has_code

    @property
    def code_strings(self):
        return [str(cell) for cell in self._cells if cell.type == "code"]

    @property
    def code_lines(self):
        return [
            line for cell in self._cells if cell.type == "code" for line in cell.content
        ]

    @property
    def num_code_cells(self) -> int:
        return sum(1 for cell in self._cells if cell.type == "code")

    @property
    def num_code_lines(self) -> int:
        return sum(len(cell.content) for cell in self._cells if cell.type == "code")

    @property
    def num_ast_children(self) -> int:
        return self._num_ast_children

    def __repr__(self) -> str:
        return f"<Segment({self._cells})>"


class Notebook:
    """Computational Notebook.

    Contains all cell data and a series of conversion utilities.
    """

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> Self:
        cells = df[df["cell_type"] == "code"]["cell_content_no_comment"].values.tolist()
        cells = [c.split("\n") for c in cells]

        return cls.from_code_cells(cells)

    @classmethod
    def from_db_result(cls, db_result) -> Self:
        segments = []
        for segment in db_result["segments"]:
            cells = []
            for cell in segment["cells"]:
                cell_type = cell["type"] if "type" in cell else "code"
                cells.append(Cell(cell_type, cell["content_no_comment"]))

            segment = Segment(cells, segment["has_code"])
            segments.append(segment)

        return cls(str(db_result["_id"]), segments)

    @classmethod
    def from_db_result_encoding(cls, db_result) -> Self:
        segments = []
        for s in db_result["encoding"]["segments"]:
            segment = Segment(
                [Cell("code", s["code"].split("\n"))], True, s["n_ast_children"]
            )
            segments.append(segment)

        return cls(str(db_result["_id"]), segments)

    @classmethod
    def from_segment_ends_ast(
        cls,
        segment_ends_ast: List[int],
        notebook_reference: Optional[Self] = None,
        code: Optional[str] = None,
    ):
        curr_index = 0
        curr_n_ast = -1

        last_offset = 0

        segments = []

        if code is None:
            code = "\n".join(notebook_reference.code_strings)

        code_bytes = code.encode("utf-8")

        tree = p.parse(code_bytes)
        cursor = tree.walk()

        cursor.goto_first_child()

        def iterate_children():
            nonlocal curr_index
            nonlocal curr_n_ast
            nonlocal last_offset

            if (
                curr_index <= len(segment_ends_ast) - 1
                and segment_ends_ast[curr_index] == curr_n_ast
            ):
                segment_code = code_bytes[last_offset : cursor.node.end_byte].decode(
                    "utf-8"
                )
                # print(curr_n_ast, segment_code)
                code_lines = segment_code.split("\n")
                segment = Segment([Cell("code", code_lines)], True)
                segments.append(segment)
                curr_index += 1
                last_offset = cursor.node.end_byte
                # print(curr_n_ast, segment)

        curr_n_ast += 1
        iterate_children()

        while cursor.goto_next_sibling():
            curr_n_ast += 1
            iterate_children()

        if last_offset < len(code_bytes) - 1:
            segment_code = code_bytes[last_offset:].decode("utf-8")
            code_lines = segment_code.split("\n")
            segment = Segment([Cell("code", code_lines)], True)
            segments.append(segment)

        if notebook_reference is not None:
            nb_id = notebook_reference.id
        else:
            nb_id = 0

        return cls(nb_id, segments)

    @classmethod
    def from_segment_ends(
        cls,
        segment_ends: List[int],
        notebook_reference: Self,
        mode: Literal["cell", "line", "ast"] = "cell",
    ) -> Self:
        segments = []
        iter_se = iter(segment_ends)
        curr_se = next(iter_se)

        if mode == "cell":
            cells = []
            for i, cell in enumerate(notebook_reference.code_cells):
                cells.append(cell)

                if i >= curr_se:
                    segment = Segment(cells, True)
                    segments.append(segment)
                    cells = []
                    try:
                        curr_se = next(iter_se)
                    except:
                        curr_se = len(notebook_reference.code_cells) - 1
                        break

        elif mode == "line":
            lines = []
            for i, line in enumerate(notebook_reference.code_lines):
                lines.append(line)

                if i >= curr_se:
                    segment = Segment([Cell("code", lines)], True)
                    segments.append(segment)
                    lines = []
                    try:
                        curr_se = next(iter_se)
                    except:
                        curr_se = len(notebook_reference.code_lines) - 1

        elif mode == "ast":
            code_lines = notebook_reference.code_lines
            ast_line_map = extract_notebook_ast_sequence(notebook_reference, p)

            # print(f"[{len(segment_ends)}] segment_ends: {segment_ends}")
            # print(f"[{len(ast_line_map)}] ast_line_map:")
            # pprint(ast_line_map)

            code_line_ends = []
            for i in segment_ends:
                try:
                    v = ast_line_map[i]["end_line_number"]
                except:
                    print(
                        f"WARN  {notebook_reference.id} predicted {i}/{len(ast_line_map)}"
                    )
                    continue

                code_line_ends.append(v)

            if len(code_line_ends) == 0 or code_line_ends[-1] < (len(code_lines) - 1):
                code_line_ends.append(len(code_lines) - 1)

            # print(f"[{len(code_line_ends)}] code_line_ends: {code_line_ends}")

            lines = []
            iter_le = iter(code_line_ends)
            curr_le = next(iter_le)

            for i, line in enumerate(code_lines):
                lines.append(line)

                if i >= curr_le:
                    segment = Segment([Cell("code", lines)], True)
                    segments.append(segment)
                    lines = []
                    try:
                        curr_le = next(iter_le)
                    except:
                        curr_le = len(code_lines) - 1

        else:
            raise ValueError("Invalid segment ends mode")

        return cls(notebook_reference.id, segments)

    @classmethod
    def from_code_cells(cls, code_cells: List[List[str]]) -> Self:
        return cls().extend_cells([Cell("code", code) for code in code_cells])

    @classmethod
    def from_db_cells(cls, id: str, cells: List[CellD]) -> Self:
        return cls(id=id).extend_cells(
            [Cell(cell["type"], cell["content"]) for cell in cells]
        )

    @classmethod
    def from_code_strings(cls, code_strings: List[str]) -> Self:
        ### v1: only 1 segment
        # return cls.from_code_cells([code_str.split("\n") for code_str in code_strings])

        ### v2: each code string is a segment
        segments = []
        for code_str in code_strings:
            segments.append(
                Segment([Cell("code", code_str.split("\n"))], has_code=True)
            )
        return cls(segments=segments)

    def __init__(
        self, id: Optional[str] = None, segments: Optional[List[Segment]] = None
    ) -> None:
        self._id = id
        self._segments: List[Segment] = segments if segments else [Segment([], False)]
        self._segment_ends: List[int] = []  # Segment ends (in number of code cells)
        self._segment_end_lines: List[
            int
        ] = []  # Segment ends (in number of code cells)
        self._compute_segment_ends()

    def _compute_segment_ends(self):
        result = [
            segment.num_code_cells for segment in self._segments if segment.has_code
        ]
        self._segment_ends = list(
            dict.fromkeys(map(lambda x: x - 1, itertools.accumulate(result)))
        )
        result = [
            segment.num_code_lines for segment in self._segments if segment.has_code
        ]
        self._segment_end_lines = list(
            dict.fromkeys(map(lambda x: x - 1, itertools.accumulate(result)))
        )

    @property
    def id(self):
        return self._id

    @property
    def segments(self) -> List[Segment]:
        return self._segments

    @property
    def segment_ends(self) -> List[int]:
        return self._segment_ends

    @property
    def segment_ends_ast(self) -> List[int]:
        n_ast_children = [
            segment.num_ast_children for segment in self._segments if segment.has_code
        ]
        return list(
            dict.fromkeys(map(lambda x: x - 1, itertools.accumulate(n_ast_children)))
        )

    @property
    def segment_end_lines(self) -> List[int]:
        return self._segment_end_lines

    @property
    def _cells(self) -> List[Cell]:
        return [cell for segment in self._segments for cell in segment.cells]

    @property
    def code_cell_end_lines(self) -> List[int]:
        result = [len(cell) for cell in self.code_cells if cell.type == "code"]
        return list(dict.fromkeys(map(lambda x: x - 1, itertools.accumulate(result))))

    def append_cell(self, cell: Cell) -> Self:
        self._segments[-1].cells.append(cell)
        return self

    def extend_cells(self, cells: List[Cell]) -> Self:
        self._segments[-1].cells.extend(cells)
        return self

    def __iter__(self):
        yield from self._cells

    def remove_code_cell_comments(self) -> Self:
        for cell in self._cells:
            try:
                cell.remove_comment()
            except:
                # print("WARN  remove comment failed")
                continue

        self._compute_segment_ends()

        return self

    def segment_id(self, line_number: int):
        for i, el in enumerate(self.segment_end_lines):
            if el >= line_number:
                return i
        return -1

    def code_cell_id(self, line_number: int):
        for i, el in enumerate(self.code_cell_end_lines):
            if el >= line_number:
                return i
        return -1

    @property
    def cells(self) -> List[Cell]:
        return self._cells

    @property
    def code_cells(self) -> List[Cell]:
        return [cell for cell in self._cells if cell.type == "code"]

    @property
    def code_lines(self) -> List[str]:
        return [line for cell in self.code_cells for line in cell.content]

    @property
    def code_strings(self) -> List[str]:
        return [str(cell) for cell in self.code_cells if cell.type == "code"]

    def __repr__(self) -> str:
        n_code_cells = sum(1 for cell in self._cells if cell.type == "code")
        n_md_cells = len(self._cells) - n_code_cells
        return (
            f"<Notebook({len(self._segments)} segments, {n_md_cells} markdown cells,"
            f" {n_code_cells} code cells)>"
        )

    def pretty_print_segments(self):
        for i, c in enumerate(self.segments):
            print(f"{i}", end="")
            for line in c.code_lines:
                print(f"\t{line}")
            print()
