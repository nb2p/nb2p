#!/usr/bin/env python
# coding: utf-8

import glob
import os
import subprocess
import time
from argparse import ArgumentParser

from nb2p import config, database


def exec_cmd_elapsed_time(path: str):
    cmd = f'source ./runOsiris.sh {path} OEC "-m best_effort -a"'

    start_time = time.time()
    proc = subprocess.Popen(cmd, shell=True, executable="/bin/bash")
    proc.communicate()

    duration = time.time() - start_time
    return duration


def profile(nb_data_chunk, ds_name, dry_run):
    """Connect to DB"""
    db = database.connect(dataset_name=ds_name)

    for nb_path in nb_data_chunk:
        nb_fname = nb_path.split("/")[-1]

        duration = exec_cmd_elapsed_time(nb_path)
        if not dry_run:
            db.profile.update_one(
                {
                    "notebook_path": nb_fname,
                    "dataset": ds_name,
                    "method": "sessrstr",
                    "model": "sessrstr",
                },
                {
                    "$set": {
                        "notebook_path": nb_fname,
                        "dataset": ds_name,
                        "method": "sessrstr",
                        "model": "sessrstr",
                        "sr_time": duration,
                    }
                },
                upsert=True,
            )

        print(f"[Worker] {nb_fname}: {duration:.4f}s")

    print(f"[Worker] done")


if __name__ == "__main__":
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

    os.chdir("Osiris")

    notebook_paths = sorted(glob.glob(str(DIRS.base / "test_nbs" / "*.ipynb")))
    if args.limit:
        notebook_paths = notebook_paths[: args.limit]

    profile(notebook_paths, args.dataset, args.dry)
