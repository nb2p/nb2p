#!/usr/bin/env python


# coding: utf-8

"""Shortcut for package import"""
import ctypes
import multiprocessing as mp
from functools import partial
import os
from typing import Union

from bson import ObjectId
from loguru import logger
from tokenizers import Tokenizer
import torch
import gc

from nb2p.astparse import parser
from nb2p.notebook import Notebook
from pkgimp import *
from dfg import CodeEncodingBuilder
from dfgtree import get_xs_model, build_dfg_tree
from transformers import RobertaConfig

"""ipmort nb2p modules"""
from nb2p import config, database
from nb2p.parallel import divide_chunks

"""Set up Directories"""
from argparse import ArgumentParser

torch.set_num_threads(1)
torch.get_num_threads()

DEVICE = torch.device("cpu")


def get_tokenizer():
    config = RobertaConfig.from_pretrained("microsoft/graphcodebert-base")
    config.num_attention_heads = 8
    config.hidden_size = 96
    config.intermediate_size = 64
    config.vocab_size = 1000
    config.num_hidden_layers = 12
    config.hidden_dropout_prob = 0.2

    tokenizer_path = os.path.join(
        "compressor", "BPE" + "_" + str(config.vocab_size) + ".json"
    )
    tokenizer = Tokenizer.from_file(tokenizer_path)

    return tokenizer


def update_encodings(
    id: Union[str, int] = "<UNKNOWN>",
    nb_data_chunk: list = [],
    ds_name: str = "",
    dry_run: bool = False,
):
    p, lang = parser()
    db, client = database.connect(dataset_name=ds_name, verbose=True)

    tokenizer = get_tokenizer()

    builder = CodeEncodingBuilder(tokenizer, None, DEVICE)

    for nb_data in nb_data_chunk:
        if "encoding" in nb_data and "n_ast_children_of_segments" in nb_data:
            # logger.debug(f"[Worker] {nb_data['_id']}: Already computed. Skipping.")
            continue

        try:
            nb = Notebook.from_db_result(nb_data)
        except Exception as e:
            logger.debug(
                f"[Worker] {nb_data['_id']}: Creating notebook failed. Reason: {e.__class__.__name__}: {e}"
            )
            continue

        try:
            encoding = build_dfg_tree(nb, p, lang, builder, with_encodings=False)
        except Exception as e:
            logger.debug(
                f"[Worker] {nb.id}: Updating encoding failed. Reason: {e.__class__.__name__}: {e}"
            )

        if not dry_run:
            db.notebook.update_one(
                {"_id": ObjectId(nb.id)},
                {
                    "$set": {
                        "encoding": encoding,
                        "n_ast_children_of_segments": encoding["segment_ends"][-1],
                    }
                },
            )
            db.notebooksegments.update_one(
                {"_id": ObjectId(nb.id)},
                {
                    "$set": {
                        "encoding": encoding,
                        "n_ast_children_of_segments": encoding["segment_ends"][-1],
                    }
                },
            )
        else:
            logger.debug(f"[Worker] {nb.id}: segment_ends={encoding['segment_ends']}")

        del nb
        del encoding

        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)

    logger.debug(f"[Worker] Chunk {id} done")

    client.close()

    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)


if __name__ == "__main__":
    mp.set_start_method("spawn")

    arg = ArgumentParser()
    arg.add_argument("-d", "--dataset", help="dataset name", required=True)
    arg.add_argument("-l", "--limit", help="first k elements", required=False, type=int)
    arg.add_argument("-j", "--job", type=int, help="number of jobs", default=8)
    arg.add_argument(
        "-D",
        "--dry",
        help="dry run (not writing DB)",
        required=False,
        action="store_true",
    )
    args = arg.parse_args()

    DIRS = config.dirs(dataset_name=args.dataset)
    DIRS.makedirs()

    db, client = database.connect(dataset_name=args.dataset)

    notebooks = database.get_notebooks(db, include_segments=True)

    if args.limit:
        notebooks = list(
            database.get_notebooks(db, include_segments=True, limit=args.limit)
        )

        update_encodings(
            nb_data_chunk=notebooks,
            ds_name=args.dataset,
            dry_run=args.dry,
        )
    else:
        """Load notebooks"""
        notebooks = list(database.get_notebooks(db, include_segments=True))
        print("[System] Notebook loaded")
        notebook_chunks = list(divide_chunks(notebooks, 1000))
        print(
            f"[System] Notebook chunks generated. Total chunks: {len(notebook_chunks)}"
        )

        p = mp.Pool(args.job)
        p.starmap(
            partial(
                update_encodings,
                ds_name=args.dataset,
                dry_run=args.dry,
            ),
            enumerate(notebook_chunks),
            chunksize=1,
        )
