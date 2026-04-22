#!/usr/bin/env python


# coding: utf-8

"""Shortcut for package import"""
import multiprocessing as mp
from functools import partial
from typing import Union

from bson import ObjectId
from loguru import logger

from pkgimp import *

"""ipmort nb2p modules"""
from nb2p import config, database
from nb2p.parallel import divide_chunks

"""Set up Directories"""
from argparse import ArgumentParser


def update_encodings(
    id: Union[str, int] = "<UNKNOWN>",
    nb_data_chunk: list = [],
    ds_name: str = "",
    dry_run: bool = False,
):
    db, client = database.connect(dataset_name=ds_name, verbose=True)

    for nb_data in nb_data_chunk:
        nb_id = str(nb_data["_id"])

        try:
            segment_ends = []
            [
                segment_ends.append(x)
                for x in nb_data["encoding"]["segment_ends"]
                if x not in segment_ends and x >= 0
            ]
        except Exception as e:
            logger.debug(
                f"[Worker] {nb_id}: Updating encoding failed. Reason: {e.__class__.__name__}: {e}"
            )

        if not dry_run:
            db.notebook.update_one(
                {"_id": ObjectId(nb_id)}, {"$set": {"segment_ends": segment_ends}}
            )
            db.notebooksegments.update_one(
                {"_id": ObjectId(nb_id)}, {"$set": {"segment_ends": segment_ends}}
            )
        else:
            logger.debug(f"[Worker] {nb_id}: segment_ends={segment_ends}")

    logger.debug(f"[Worker] Chunk {id} done")

    client.close()


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
