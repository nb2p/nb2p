import gc
import glob
import os
import pathlib
import pickle
import traceback
import warnings
from typing import List, Union

import lz4.frame
import pymongo.database
import torch
from nb2p.dfgtree.dfg import CodeEncodingBuilder
from nb2p.dfgtree.dfgtree import DFGTree, preprocess
from torch import nn
from tqdm import tqdm
from tree_sitter import Language, Parser

from nb2p import fileop
from nb2p.notebook import Notebook


def to_files(
    db: pymongo.database.Database,
    builder: CodeEncodingBuilder,
    parser: Parser,
    lang: Language,
    out_dir: pathlib.Path,
    ids: List[int],
    fname: str,
    report_interval=10,
    full_check: bool = False,
):
    n_completed = 0
    ignored = []
    pbar = tqdm(enumerate(ids), total=len(ids))

    for i, nb_id in pbar:
        fpath = out_dir / f"{fname}-{nb_id}-tree.lz4"

        if os.path.exists(fpath):
            n_completed += 1

            if full_check and n_completed > 1000:
                warnings.warn(
                    "The current execution has overlapped with another running. "
                    "Stop to avoid race condition",
                    RuntimeWarning,
                )
                break

            continue

        nb_data = db.notebooksegments.find_one({"_id": nb_id})
        if nb_data is None:
            continue

        nb = Notebook.from_db_result(nb_data)
        try:
            pr = preprocess(nb, parser, lang)
            tree = DFGTree(input=pr, builder=builder).build()
        except Exception as e:
            ignored.append(nb_id)
            print(nb_id, e)
            continue

        with lz4.frame.open(fpath, "wb") as f:
            f.write(pickle.dumps(tree))

        pbar.set_postfix(
            {
                "n_ignored": len(ignored),
                "fd_len": len(tree["func_defs"]),
                "n_seg": len(tree["segments"]),
            }
        )

        if i % report_interval == (report_interval - 1):
            # torch.cuda.empty_cache()
            gc.collect()
            with open(out_dir / ".cursor", "w") as cursor_f:
                cursor_f.write(f"{i+1}/{len(ids)}\n")

        n_completed = 0

    return ignored


def sort_chunks(glob_pattern: str):
    files = glob.glob(glob_pattern)
    files.sort(key=lambda x: x.split(".")[-2].split("-")[-2])
    return files


def merge_chunks(glob_pattern: str, dest_path: Union[str, bytes, os.PathLike]):
    result = []

    sorted_chunks = sort_chunks(glob_pattern)

    for chunk_file_path in tqdm(sorted_chunks, total=len(sorted_chunks)):
        result.append(fileop.read_lz4(chunk_file_path))
        # print(f"[merge_chunks] current chunk length: {len(result)}")

    print(f"[merge_chunks] done. total chunk length: {len(result)}")

    fileop.dump_lz4(dest_path, result)
    print(f"[merge_chunks] dumped to: {dest_path}")
