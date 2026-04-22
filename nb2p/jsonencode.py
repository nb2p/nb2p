"""JSON encoder."""

import json
from collections.abc import Callable
from typing import Any

import numpy as np
from bson.objectid import ObjectId


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, ObjectId):
            return str(obj)
        return super(NpEncoder, self).default(obj)
