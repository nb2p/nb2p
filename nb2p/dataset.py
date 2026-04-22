"""Dataset analyzer and utilities."""

import glob
import json
import os
import sys
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import List, Optional, Tuple, Union

if sys.version_info >= (3, 11):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

import bson
# import fasttext
import pandas as pd
import peewee
from IPython.display import Image, Markdown, display  # type:ignore
from loguru import logger
from pygments import highlight
from pygments.formatters import ImageFormatter
from pygments.lexers import PythonLexer
from pymongo.database import Database as PyMongoDB
from tqdm import tqdm

from nb2p import codeop, config

from . import database


class CellD(TypedDict):
    _id: bson.ObjectId
    notebook_id: bson.ObjectId
    cell_id: int
    type: str
    content: List[str]
    content_no_comment: Optional[List[str]]


class NotebookD(TypedDict):
    _id: bson.ObjectId
    notebook_id: int
    cells: List[CellD]
    can_parse: bool


class SegmentD(TypedDict):
    _id: bson.ObjectId
    notebook_id: int
    segment_id: int
    prompts: List[str]
    code_cells: List[CellD]
    has_code: bool


class NbSegmentD(TypedDict):
    prompts: List[str]
    code_lines: List[str]


class NbSegmentsD(TypedDict):
    notebook_id: int
    segments: List[SegmentD]


def load_fasttext_model():
    model = fasttext.load_model(config.parse_path(config.value("fasttext.path")))
    # print(model.predict('الشمس تشرق', k=2))  # top 2 matching languages
    return model


class ScriptsStat(TypedDict):
    num_original: int
    num_valid: int
    num_prompted: int
    num_unprompted: int


# class BaseScriptsBuilder(metaclass=ABCMeta):
#     def __init__(
#         self,
#         notebook_dir: str,
#         n_try: Optional[int] = None,
#         lang_model: Optional[fasttext.FastText._FastText] = None,
#     ) -> None:
#         self._notebook_dir = notebook_dir
#         self._n_try = n_try
#         self._lang_model = load_fasttext_model() if lang_model is None else lang_model
#         self._stat: ScriptsStat = {
#             "num_original": -1,
#             "num_valid": -1,
#             "num_prompted": -1,
#             "num_unprompted": -1,
#         }

#     @abstractmethod
#     def _get_unique_columns(self) -> List[str]:
#         pass

#     @abstractmethod
#     def _parse_unique_columns_dict(self, pattern_str: str) -> OrderedDict:
#         pass

#     def _get_id(self, unique_columns_dict: OrderedDict):
#         return "_".join(map(str, unique_columns_dict.values()))

#     def _build_setup(self):
#         pd.options.mode.chained_assignment = None  # type:ignore

#     def build(self) -> Tuple[pd.DataFrame, pd.DataFrame, ScriptsStat]:
#         self._build_setup()
#         ipynb_files = glob.glob(os.path.join(self._notebook_dir, "*.ipynb"))

#         scripts_df = pd.DataFrame(
#             columns=self._get_unique_columns()
#             + ["cell_id", "cell_type", "cell_content"]
#         )
#         unprompted_df = pd.DataFrame(
#             columns=self._get_unique_columns()
#             + ["cell_id", "cell_type", "cell_content"]
#         )

#         scripts_dfs = []
#         unprompted_dfs = []

#         self._stat["num_original"] = self._n_try if self._n_try else len(ipynb_files)

#         pbar = tqdm(enumerate(ipynb_files), total=self._stat["num_original"])
#         ignored = []
#         num_non_english = 0

#         for i, fpath in pbar:
#             if self._n_try is not None and i >= self._n_try:
#                 break

#             pattern = fpath.split("/")[-1].split(".")[-2]
#             unique_columns_dict = self._parse_unique_columns_dict(pattern)

#             # user_name, current_url_slug = pattern.split("_")

#             with open(fpath, "r") as f:
#                 try:
#                     notebook = json.loads(f.read())
#                 except json.JSONDecodeError:
#                     ignored.append(self._get_id(unique_columns_dict))
#                     # print(f'WARN: Cannot parse JSON for {i}')
#                     continue

#             try:
#                 iter(notebook)
#             except TypeError:
#                 ignored.append(self._get_id(unique_columns_dict))
#                 # print(
#                 #     f"WARN: Not valid ipynb file for {self._get_id(unique_columns_dict)}"
#                 # )
#                 continue

#             if (not notebook) or (not "cells" in notebook) or (not notebook["cells"]):
#                 ignored.append(self._get_id(unique_columns_dict))
#                 # print(
#                 #     f"WARN: Not valid ipynb format 1 for {self._get_id(unique_columns_dict)}"
#                 # )
#                 continue

#             df = pd.DataFrame(notebook["cells"])
#             if ("source" not in df.columns) or ("cell_type" not in df.columns):
#                 ignored.append(self._get_id(unique_columns_dict))
#                 # print(
#                 #     f"WARN: Not valid ipynb format 2 for {self._get_id(unique_columns_dict)}"
#                 # )
#                 continue

#             df["source"] = df["source"].apply(
#                 lambda x: "".join(x) if isinstance(x, list) else x
#             )

#             df_text = df.query("cell_type == 'markdown'")
#             df_text["lang"] = df_text["source"].apply(
#                 lambda x: (
#                     str(
#                         self._lang_model.predict(
#                             str(x[:50].replace("\n", "").replace("#", "")), k=1
#                         )[0][
#                             0
#                         ]  # type: ignore
#                     )
#                     if isinstance(x, str)
#                     else ""
#                 )
#             )
#             df_non_eng = df_text[df_text["lang"] != "__label__en"]

#             # display(df_non_eng)
#             # display(df_text)

#             if len(df_non_eng) > (0.5 * len(df_text)):
#                 # print('non-English notebook')
#                 ignored.append(self._get_id(unique_columns_dict))
#                 num_non_english += 1
#                 continue

#             df_comp_pt = df.query("cell_type == 'markdown'")
#             df_comp_pt = df_comp_pt[
#                 df_comp_pt["source"].apply(
#                     lambda x: x.startswith("#") if isinstance(x, str) else False
#                 )
#             ]
#             if len(df_comp_pt.index) == 0:
#                 # print('does not have component prompt')
#                 continue

#             # df["user_name"] = user_name
#             # df["current_url_slug"] = current_url_slug

#             for k, v in unique_columns_dict.items():
#                 df[k] = v

#             df["cell_content"] = df["source"]
#             df = df[df["cell_content"] != ""]
#             df.drop(columns=["source"], inplace=True)

#             df.reset_index(names=["cell_id"], inplace=True)
#             df = df[
#                 self._get_unique_columns()
#                 + [
#                     "cell_id",
#                     "cell_type",
#                     "cell_content",
#                 ]
#             ]

#             # display(df)

#             df_self_merged = df.merge(df, how="cross", suffixes=("_2", ""))
#             df_self_merged = df_self_merged.query(
#                 "(cell_id_2 > cell_id) and (cell_type == 'code') and (cell_type_2 =="
#                 " 'markdown')"
#             )
#             df_self_merged = df_self_merged[
#                 df_self_merged["cell_content_2"].apply(
#                     lambda x: x.startswith("#") if isinstance(x, str) else False
#                 )
#             ]

#             if len(df_self_merged) == 0:
#                 # unprompted_df = pd.concat([unprompted_df, df])
#                 unprompted_dfs.append(df)
#                 self._stat["num_unprompted"] += 1
#             else:
#                 # Has markdown cell below code cell (possibly a prompt for next cells)
#                 # Assumption: has component prompt in markdown description

#                 # print('has component prompt')
#                 # display(df)

#                 scripts_dfs.append(df)
#                 # scripts_df = pd.concat([scripts_df, df])
#                 self._stat["num_prompted"] += 1

#             pbar.set_postfix(
#                 {
#                     "num_prompted": self._stat["num_prompted"],
#                     "num_unprompted": self._stat["num_unprompted"],
#                     "num_ignored": len(ignored),
#                     "num_non_english": num_non_english,
#                 }
#             )

#         scripts_df = pd.concat(scripts_dfs)
#         unprompted_df = pd.concat(unprompted_dfs)

#         scripts_df.reset_index(drop=True, inplace=True)

#         self._stat["num_valid"] = self._stat["num_original"] - len(ignored)

#         return scripts_df, unprompted_df, self._stat


def render_notebook_df(notebook_df: pd.DataFrame):
    """Render a notebook.

    Args:
        notebook_df: A dataframe with columns ["cell_type", "cell_content"].
    """
    for i, cell in enumerate(notebook_df.to_dict(orient="records")):
        if cell["cell_type"] == "markdown":
            display(Markdown(cell["cell_content"]))
        else:
            image = highlight(cell["cell_content"], PythonLexer(), ImageFormatter())
            display(Image(image))


def render_notebook_dict(notebook: NotebookD):
    """Render a notebook.

    Args:
        notebook: A notebook.
    """
    for i, cell in enumerate(notebook["cells"]):
        content = "\n".join(cell["content"])
        if cell["type"] == "markdown":
            display(Markdown(content))
        else:
            image = highlight(content, PythonLexer(), ImageFormatter())
            display(Image(image))


def render_python_code(code: str):
    return highlight(code, PythonLexer(), ImageFormatter())


def get_segments(notebook: NotebookD, include_content=False) -> List[SegmentD]:
    """Convert a notebook to a list of segments.

    Args:
        notebook: A notebook dict.
        include_content: Whether to include content or append ID only.

    Returns:
        A list of segments.
    """
    segments = []
    curr_cells: List[Union[CellD, bson.ObjectId]] = []

    has_code = False
    for cell in notebook["cells"]:
        if cell["type"] == "markdown" and cell["content"]:
            for line in cell["content"]:
                if line.startswith("#") and has_code:
                    # next segment
                    segments.append({"cells": curr_cells, "has_code": has_code})
                    curr_cells = []
                    has_code = False
                    continue
            (
                # Code in middle. Add code indicator to current segment.
                curr_cells.append(cell)
                if include_content
                else curr_cells.append(cell["_id"])
            )
            continue

        if cell["type"] == "code":
            (
                curr_cells.append(cell)
                if include_content
                else curr_cells.append(cell["_id"])
            )
            has_code = True

    if len(curr_cells) > 0:
        # final segment
        segments.append({"cells": curr_cells, "has_code": has_code})

    return segments


def to_segments_df(notebook: pd.DataFrame) -> List[SegmentD]:
    """Convert a notebook (in Pandas DataFrame format) to a list of segments

    Args:
        notebook (pd.DataFrame): A dataframe with columns ["cell_type", "cell_content"]

    Returns:
        List[Segment]: A list of segments
    """
    segments = []

    curr_prompts = []
    curr_cells = []
    for r in notebook.to_dict("records"):
        if r["cell_type"] == "markdown" and isinstance(r["cell_content"], str):
            for l in r["cell_content"].split("\n"):
                if l.startswith("#"):
                    if len(curr_cells) > 0:
                        # next segment
                        segments.append({"prompts": curr_prompts, "cells": curr_cells})
                        curr_prompts = [r["cell_content"]]
                        curr_cells = []
                        continue
                    elif len(curr_prompts) == 0:
                        pass
                    else:
                        # Code in middle. Add code indicator to current segment.
                        curr_prompts.append("$CODE$")
                        # the program will continue to execute to add this line
                        # to the current segment

                # a segment with multiple markdown cells for prompting
                curr_prompts.append(l)
                continue

        if r["cell_type"] == "code":
            curr_cells.append(r["cell_content"])

    if len(curr_cells) > 0:
        # final segment
        segments.append({"prompts": curr_prompts, "cells": curr_cells})

    return segments


def update_no_comment_for_code_cells(db, dry_run=False):
    for nb in database.get_notebooks(db):
        can_parse = True

        for cell in nb["cells"]:
            if cell["type"] == "code":
                try:
                    cell["content_no_comment"] = codeop.remove_comments_and_docstrings(
                        "\n".join(cell["content"])
                    ).split("\n")
                except:
                    cell["content_no_comment"] = None
                    can_parse = False

                if not dry_run:
                    try:
                        db.cell.update_one(
                            {"_id": cell["_id"]},
                            {
                                "$set": {
                                    "content_no_comment": cell["content_no_comment"]
                                }
                            },
                        )
                    except Exception as e:
                        cell["content_no_comment"] = None
                        can_parse = False
                        logger.warning(f"Cannot add {nb['_id']} to DB: {e}")

        if not dry_run:
            db.notebook.update_one(
                {"_id": nb["_id"]}, {"$set": {"can_parse": can_parse}}
            )
        else:
            print(f"[DRY_RUN] updating {nb['_id']}: can_parse={can_parse}")


def update_n_code_lines(db):
    for nb in database.get_notebooks(db):
        n_code_lines = 0
        n_code_lines_no_comment = 0

        for cell in nb["cells"]:
            if cell["type"] == "code":
                n_code_lines += len(cell["content"])

                if "content_no_comment" in cell and cell["content_no_comment"]:
                    n_code_lines_no_comment += len(cell["content_no_comment"])

        db.notebook.update_one(
            {"_id": nb["_id"]},
            {
                "$set": {
                    "n_code_lines": n_code_lines,
                    "n_code_lines_no_comment": n_code_lines_no_comment,
                }
            },
        )


def insert_into_db(db: PyMongoDB, df: pd.DataFrame):
    for nb in df.to_dict(orient="records"):
        cells: List[CellD] = nb["cells"]
        del nb["cells"]

        result = db.notebook.insert_one(nb)
        object_id = result.inserted_id

        for i, cell in enumerate(cells):
            cell["notebook_id"] = object_id
            cell["cell_id"] = i
        result = db.cell.insert_many(cells)
        cell_ids = result.inserted_ids

        db.notebook.update_one({"_id": object_id}, {"$set": {"cells": cell_ids}})


def materialize(db: PyMongoDB):
    """Materialize notebook segments collection.

    This will create a separate `notebooksegment` collection to accelerate
    the queries.

    Args:
        db: Database connection.
    """
    db.notebook.aggregate(
        [
            {"$match": {"prompted": True}},
            {
                "$lookup": {
                    "from": "segment",
                    "localField": "segments",
                    "foreignField": "_id",
                    "as": "segments",
                }
            },
            {"$unwind": {"path": "$segments"}},
            {
                "$lookup": {
                    "from": "cell",
                    "localField": "segments.cells",
                    "foreignField": "_id",
                    "as": "segments.cells",
                }
            },
            {
                "$group": {
                    "_id": "$_id",
                    "notebook_id": {"$first": "$notebook_id"},
                    "user_name": {"$first": "$user_name"},
                    "current_url_slug": {"$first": "$current_url_slug"},
                    "prompted": {"$first": "$prompted"},
                    "can_parse": {"$first": "$can_parse"},
                    "segments": {"$push": "$segments"},
                }
            },
            {"$sort": {"_id": 1}},
            {"$merge": {"into": "notebooksegments", "whenMatched": "replace"}},
        ]
    )


def write_segments_to_db(db: PyMongoDB):
    for nb in database.get_notebooks(db, {"prompted": True}):
        object_id = nb["_id"]

        segments = get_segments(nb)

        for i, segment in enumerate(segments):
            segment["notebook_id"] = object_id
            segment["segment_id"] = i
        result = db.segment.insert_many(segments)
        cell_ids = result.inserted_ids

        db.notebook.update_one({"_id": object_id}, {"$set": {"segments": cell_ids}})
