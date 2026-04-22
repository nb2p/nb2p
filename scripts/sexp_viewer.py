import os
import re
import subprocess
import sys

from tree_sitter import Language, Parser

PY_LANGUAGE = Language("../build/tree-sitter-python.so", "python")

parser = Parser()
parser.set_language(PY_LANGUAGE)

# with open("../_example_notebook/kgtorrent-99071.py") as f:
#     src = f.read().encode("utf-8")

# tree = parser.parse(src)
# sexp = tree.root_node.

FILE_PATH = f"../_example_notebook/kgtorrent-{sys.argv[1]}.py"

with open(FILE_PATH) as f:
    lines = f.read().split("\n")

process = subprocess.Popen(
    f"tree-sitter parse {FILE_PATH}",
    shell=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

# wait for the process to terminate
out, err = process.communicate()
errcode = process.returncode

sexp = out.decode()

if errcode != 0:
    sexp = sexp[:-1]


def replace_points_by_identifiers(match):
    start_line = int(match.group(1))
    start_char = int(match.group(2))
    end_line = int(match.group(3))
    end_char = int(match.group(4))

    # print(start_line, start_char, end_line, end_char)

    if start_line == end_line:
        return f"{match[0]}: '{lines[start_line][start_char:end_char]}'"
    else:
        return match.group(0)


sexp_identifiers = re.sub(
    r"\[([^,]+), ([^\]]+)\] - \[([^,]+), ([^\]]+)\]",
    replace_points_by_identifiers,
    sexp,
)

print(sexp_identifiers)
