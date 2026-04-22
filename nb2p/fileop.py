"""File operations."""

import io
import json
import os
import pickle
from pathlib import Path
from typing import Optional, Union


def write_pickle(obj: object, path: Union[str, bytes, os.PathLike]):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(path: Union[str, bytes, os.PathLike]):
    with open(path, "rb") as f:
        return pickle.load(f)


def write_json(
    obj: object,
    path: Union[str, bytes, os.PathLike],
    cls: Optional[type[json.JSONEncoder]] = None,
):
    with open(path, "w") as f:
        json.dump(obj, f, cls=cls)  # type:ignore


def read_json(path: Union[str, bytes, os.PathLike]):
    with open(path) as f:
        return json.load(f)


def read_joblib(path: Union[str, bytes, os.PathLike]):
    from joblib import load

    return load(path)


def write_joblib(obj: object, path: Union[str, bytes, os.PathLike]):
    from joblib import dump

    dump(obj, path)


def read_lz4(filepath: Union[str, bytes, os.PathLike]):
    import lz4.frame

    with lz4.frame.open(filepath, "rb") as f:
        return pickle.loads(f.read())  # type:ignore


def dump_lz4(filepath: Union[str, bytes, os.PathLike], obj: object):
    import lz4.frame

    with lz4.frame.open(filepath, "wb") as f:
        f.write(pickle.dumps(obj))  # type:ignore
