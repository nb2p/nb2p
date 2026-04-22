"""Database connector for notebook retrival."""

from typing import Optional, Union

import bson
from loguru import logger
from pymongo.database import Database
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

from nb2p import config
from nb2p.config import default_config as conf


def connect(
    database_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    verbose: bool = False,
):
    addr, port = conf["database"]["addr"], conf["database"]["port"]

    client = MongoClient(
        f"mongodb://{addr}:{port}",
        server_api=ServerApi(
            "1"
        ),  # Set the Stable API version when creating a new client,
        timeoutMS=None,
        socketTimeoutMS=None,
        connectTimeoutMS=None,
        serverSelectionTimeoutMS=21600000,
    )

    if dataset_name is not None:
        database_name = config.value(keys=["nbsegment", dataset_name, "db_name"])

    if database_name is None:
        raise ValueError("Did not specify database name and dataset name")

    db = client[database_name]
    db.command("ping")  # Send a ping to confirm a successful connection
    if verbose:
        print(
            f"Pinged to database {database_name}. You successfully connected to"
            " MongoDB!"
        )

    return db, client


def get_notebooks(
    db: Database,
    filter: dict = {},
    include_segments: bool = False,
    limit: int = 2**31 - 1,
):
    if include_segments:
        querying_db = db.notebooksegments
        return querying_db.find(filter).limit(limit)
    else:
        querying_db = db.notebook
        return querying_db.aggregate(
            [
                {
                    "$lookup": {
                        "from": "cell",
                        "localField": "cells",
                        "foreignField": "_id",
                        "pipeline": [
                            {"$sort": {"cell_id": 1}},
                        ],
                        "as": "cells",
                    }
                },
                {"$match": filter},
                {"$limit": limit},
            ]
        )


def get_notebook(
    db: Database, id: Union[bson.ObjectId, int], include_segments: bool = False
):
    if include_segments:
        querying_db = db.notebooksegments
    else:
        querying_db = db.notebook

    if isinstance(id, int):
        return querying_db.find_one({"notebook_id": id})
    else:
        return querying_db.find_one({"_id": id})


def remove_notebook(db: Database, id: bson.ObjectId):
    nb = db.notebook.find_one({"_id": id})
    if not nb:
        raise KeyError(f"Notebook {id} not found")

    for c_id in nb["cells"]:
        result = db.cell.delete_one({"_id": c_id})
        if result.deleted_count == 0:
            logger.warning(f"Cell {id} not found. Not removing")

    logger.debug("Cells removed")

    if "notebooksegments" not in db.list_collection_names():
        logger.warning(f"Collection notebooksegments not found. Skipped")

    result = db.notebooksegments.delete_one({"_id": id})
    if result.deleted_count == 0:
        logger.warning(f"Notebooksegments {id} not found. Not removing")
    else:
        logger.debug("Notebooksegments removed")

    result = db.segment.delete_many({"notebook_id": id})
    if result.deleted_count == 0:
        logger.warning(f"Segments not found. Not removing")
    else:
        logger.debug(f"Segments removed. Count: {result.deleted_count}")


def reset(db: Database):
    db.notebook.drop()
    db.cell.drop()
    db.segment.drop()
    db.notebooksegments.drop()

    db.cell.create_index(["notebook_id", "cell_id"], unique=True)
    db.segment.create_index(["notebook_id", "segment_id"], unique=True)
