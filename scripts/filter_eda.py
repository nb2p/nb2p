#!/usr/bin/env python
# coding: utf-8

import multiprocessing as mp
from argparse import ArgumentParser
from functools import partial

from bson.objectid import ObjectId
from pkgimp import *
from tree_sitter import Language, Node, Parser, Tree

from nb2p import config, database
from nb2p.astparse import node_to_code_token
from nb2p.notebook import Notebook
from nb2p.parallel import divide_chunks


def parse_import(node: Node, code_lines: List[str]):
    var_node = None

    if node.type == "import_statement":
        var_node = node.child_by_field_name("name")
        if var_node and var_node.type == "alias_import":
            var_node = var_node.child_by_field_name("name")

    if node.type == "import_from_statement":
        var_node = node.child_by_field_name("module_name")

    if var_node:
        var_name = node_to_code_token(var_node, code_lines)
        yield var_name

    else:
        return


def traverse_get_imports(tree: Tree, code_lines: List[str]):
    """Depth-first Search of tree nodes."""
    cursor = tree.walk()

    reached_root = False
    while reached_root == False:
        """TT: Yield current node"""
        # yield cursor.node
        curr_node = cursor.node
        yield from parse_import(curr_node, code_lines)

        """TT: Yield first child (Depth += 1)"""
        if cursor.goto_first_child():
            continue

        """TT: Yield other children (Same depth)"""
        if cursor.goto_next_sibling():
            continue

        """TT: Returning to parent. If the last child, return again"""
        retracing = True
        while retracing:
            if not cursor.goto_parent():
                """TT: Depth = 1, i.e., the moment from root_node_children[-1] to root"""
                retracing = False
                reached_root = True

            if cursor.goto_next_sibling():
                """TT: Have other children at parent depth. Go in"""
                retracing = False


def filter_notebook_imports(
    nb: Notebook, should_import: List[str], should_not_import: List[str], parser: Parser
) -> bool:
    parse_tree = parser.parse("\n".join(nb.code_lines).encode())
    imports = traverse_get_imports(parse_tree, nb.code_lines)

    has_import = set()
    for imp in imports:
        for sni in should_not_import:
            if sni in imp:
                return False

        for si in should_import:
            if si in imp:
                has_import.add(si)

    for si in should_import:
        if not si in has_import:
            return False

    return True


def filter_eda(i, nb_data_chunk, ds_name, dry_run):
    """Connect to DB"""
    db = database.connect(dataset_name=ds_name)

    """Set up parser"""
    parser = Parser()
    parser.set_language(Language("../build/tree-sitter-python.so", "python"))

    num_eda = 0

    for nb_data in nb_data_chunk:
        try:
            nb = Notebook.from_db_cells(str(nb_data["_id"]), nb_data["cells"])
        except:
            continue

        is_eda = filter_notebook_imports(
            nb, ["pandas"], ["tensorflow", "torch", "sklearn"], parser=parser
        )
        if is_eda:
            num_eda += 1
            if dry_run:
                print(f"[Worker] Found EDA notebook: {nb.id}")

        if not dry_run:
            db.notebook.update_one(
                {"_id": ObjectId(nb.id)},
                {"$set": {"is_eda": is_eda}},
            )
            db.notebooksegments.update_one(
                {"_id": ObjectId(nb.id)},
                {"$set": {"is_eda": is_eda}},
            )

    print(f"[Worker] Chunk {i} done. Number of EDA notebooks: {num_eda}")
    return num_eda


if __name__ == "__main__":
    mp.set_start_method("spawn")

    arg = ArgumentParser()
    arg.add_argument("-d", "--dataset", help="dataset name", required=True)
    arg.add_argument("-l", "--limit", help="first k elements", required=False, type=int)
    arg.add_argument(
        "-D",
        "--dry",
        help="dry run (not writing DB)",
        required=False,
        action="store_true",
    )
    args = arg.parse_args()

    DIRS = config.dirs(args.dataset)
    DIRS.makedirs()

    db = database.connect(dataset_name=args.dataset)

    if args.limit:
        notebooks = list(
            database.get_notebooks(db, include_segments=False, limit=args.limit)
        )

        filter_eda(
            0,
            notebooks,
            ds_name=args.dataset,
            dry_run=args.dry,
        )
    else:
        notebooks = list(database.get_notebooks(db, include_segments=False))
        print("[System] Notebook loaded")
        notebook_chunks = list(divide_chunks(notebooks, 1000))
        print(
            f"[System] Notebook chunks generated. Total chunks: {len(notebook_chunks)}"
        )

        p = mp.Pool(92)
        p.starmap(
            partial(
                filter_eda,
                ds_name=args.dataset,
                dry_run=args.dry,
            ),
            enumerate(notebook_chunks),
            chunksize=1,
        )
