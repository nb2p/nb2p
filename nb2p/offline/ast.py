from typing import Union

from bson.objectid import ObjectId
from loguru import logger
from tree_sitter import Parser

from nb2p import database
from nb2p.astparse import parser
from nb2p.notebook import Notebook


def get_num_ast_children(nb: Notebook, parser: Parser):
    parse_tree = parser.parse("\n".join(nb.code_lines).encode())
    cursor = parse_tree.walk()

    cursor.goto_first_child()

    num_ast_children = 1
    while cursor.goto_next_sibling():
        num_ast_children += 1

    return num_ast_children


def update_num_ast_children(
    id: Union[str, int] = "<UNKNOWN>",
    nb_data_chunk: list = [],
    ds_name: str = "",
    dry_run: bool = False,
):
    """Connect to DB"""
    db, client = database.connect(dataset_name=ds_name)

    """Set up parser"""
    p, lang = parser()

    for nb_data in nb_data_chunk:
        try:
            nb = Notebook.from_db_cells(str(nb_data["_id"]), nb_data["cells"])
            nb.remove_code_cell_comments()
        except Exception as e:
            logger.debug(
                f"[Worker] {nb_data['_id']}: Creating notebook failed. Reason: {e.__class__.__name__}: {e}"
            )
            continue

        num_ast_children = get_num_ast_children(nb, parser=p)
        if not dry_run:
            db.notebook.update_one(
                {"_id": ObjectId(nb.id)},
                {"$set": {"n_ast_children": num_ast_children}},
            )
            db.notebooksegments.update_one(
                {"_id": ObjectId(nb.id)},
                {"$set": {"n_ast_children": num_ast_children}},
            )
        else:
            logger.debug(f"[Worker] {nb.id}: {num_ast_children} AST children")

    logger.debug(f"[Worker] Chunk {id} done")


def update_num_ast_children_in_profile(db, nb_id, dry_run=False):
    """Set up parser"""
    p = parser()

    try:
        nb = Notebook.from_db_result(
            database.get_notebook(db, ObjectId(nb_id), include_segments=True)
        )
    except:
        print(f"[Worker] {nb_id}: failed")
        return

    num_ast_children = get_num_ast_children(nb, parser=p)
    if not dry_run:
        modified_count = db.profile.update_many(
            {"notebook_id": ObjectId(nb_id)},
            {"$set": {"n_ast_children": num_ast_children}},
            upsert=False,
        ).modified_count

        if modified_count > 0:
            print(
                f"[Worker] {nb.id}: {num_ast_children} AST children. Updated {modified_count} documents"
            )
    else:
        print(f"[Worker] {nb.id}: {num_ast_children} AST children")


def update_num_ast_children_in_profile_2(db, nb_id, dry_run=False):
    # try:
    #     nb_data = database.get_notebook(db, ObjectId(nb_id), include_segments=True)
    #     if not nb_data:
    #         raise ValueError("no such notebook")
    # except Exception as e:
    #     print(f"[Worker] {nb_id}: failed. reason: {e}")
    #     return

    # if 'n_ast_children_of_segments' not in nb_data:
    #     print(f"[Worker] {nb_id}: failed. reason: no 'n_ast_children_of_segments'")
    #     return

    # num_ast_children = nb_data['n_ast_children_of_segments']

    try:
        num_ast_children = sum(
            cell_data["n_ast_children_of_cell"]
            for cell_data in db.cell.find({"notebook_id": ObjectId(nb_id)})
            if "n_ast_children_of_cell" in cell_data
        )
    except Exception as e:
        print(f"[Worker] {nb_id}: failed. reason: {e}")
        return

    if not dry_run:
        modified_count = db.profile.update_many(
            {"notebook_id": ObjectId(nb_id)},
            {"$set": {"n_ast_children": num_ast_children}},
            upsert=False,
        ).modified_count

        if modified_count > 0:
            print(
                f"[Worker] {nb_id}: {num_ast_children} AST children. Updated {modified_count} documents"
            )
    else:
        print(f"[Worker] {nb_id}: {num_ast_children} AST children")
