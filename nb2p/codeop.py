"""Code operations."""

import io
from re import A
import tokenize
from typing import List

import numpy as np
from tree_sitter import Parser


def code_cells_split_newline(code_cells: List[str]) -> List[List[str]]:
    return [c.split("\n") for c in code_cells if c]  # Add `if c` to avoid None cells


def code_cells_to_code_lines(code_cells: List[List[str]]) -> List[str]:
    return [cl for cc in code_cells for cl in cc]


def segment_ends_to_binary(segment_ends):
    result = np.zeros((segment_ends[-1] + 1), dtype=np.int32)
    for se in segment_ends:
        result[se] = 1
    return list(result)


def binary_to_segment_ends(binary):
    result = []
    for i, binary in enumerate(binary):
        if binary > 0.5:
            result.append(i)
    return result


def remove_comments_and_docstrings(source: str) -> str:
    io_obj = io.StringIO(source)
    out = ""
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0

    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type = tok[0]
        token_string = tok[1]
        start_line, start_col = tok[2]
        end_line, end_col = tok[3]
        ltext = tok[4]

        if start_line > last_lineno:
            last_col = 0

        if start_col > last_col:
            out += " " * (start_col - last_col)

        if token_type == tokenize.COMMENT:
            pass
        elif token_type == tokenize.STRING:
            if prev_toktype != tokenize.INDENT:
                if prev_toktype != tokenize.NEWLINE:
                    if start_col > 0:
                        out += token_string
        else:
            out += token_string

        prev_toktype = token_type
        last_col = end_col
        last_lineno = end_line

    return "\n".join(l for l in out.splitlines() if l.strip())


def get_num_ast_children(code: str, parser: Parser) -> int:
    parse_tree = parser.parse(code.encode())
    cursor = parse_tree.walk()

    cursor.goto_first_child()

    num_ast_children = 1
    while cursor.goto_next_sibling():
        num_ast_children += 1

    return num_ast_children



if __name__ == "__main__":
    a_code_cells = [
        "This is the first line\nThis is the second\nThe third",
        "Starting a new cell",
        "New cell again\nFinishing the cell",
    ]

    print(code_cells_split_newline(a_code_cells))
    print(code_cells_to_code_lines(code_cells_split_newline(a_code_cells)))
