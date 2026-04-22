#!/usr/bin/env python
# coding: utf-8

"""Shortcut for package import"""
import multiprocessing as mp
from functools import partial

from bson.objectid import ObjectId
from pkgimp import *
from tree_sitter import Language, Node, Parser, Tree

"""Allow importing common module"""
COMMON_DIR = "../.."
if os.path.abspath(COMMON_DIR) not in sys.path:
    sys.path.append(os.path.abspath(COMMON_DIR))

"""ipmort nb2p modules"""
from nb2p import database
from nb2p.astparse import node_to_code_token
from nb2p.notebook import Notebook
from nb2p.parallel import divide_chunks

"""Import specific modules"""


"""Set up Directories"""
from argparse import ArgumentParser


def view_notebook(nb_data, remove_comment):
    """Connect to DB"""
    try:
        nb = Notebook.from_db_result(nb_data)
    except:
        print(f"FATAL  Notebook parse failed")
        return

    if remove_comment:
        nb.remove_code_cell_comments()

    for cell in nb.cells:
        for line in cell.content:
            if cell.type == "code":
                print(f">   {line}")
            else:
                print(f"M   {line}")

        print()


if __name__ == "__main__":
    arg = ArgumentParser()
    arg.add_argument("-d", "--dataset", help="dataset name", required=True)
    arg.add_argument("-a", "--addr", help="database address", required=True)
    arg.add_argument("-i", "--id", help="notebook id", required=True)
    arg.add_argument(
        "-C",
        "--remove-comment",
        help="remove comment",
        required=False,
        action="store_true",
    )
    args = arg.parse_args()

    if args.dataset == "kgtorrent":
        from nb2p.config import KGTDir as DIRS

        db_name = "nb2p-kgt"
    elif args.dataset == "gh17":
        from nb2p.config import GH17Dir as DIRS

        db_name = "nb2p-gh17"
    else:
        raise ValueError("No such dataset")

    mp.set_start_method("spawn")

    DIRS.makedirs()

    """Connect to DB"""
    db = database.connect(db_name, addr=args.addr)
    nb = database.get_notebook(db, ObjectId(args.id), include_segments=True)

    view_notebook(nb, remove_comment=args.remove_comment)
