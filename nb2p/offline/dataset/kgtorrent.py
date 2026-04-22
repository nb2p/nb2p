from collections import OrderedDict
from typing import List

import pandas as pd

# from nb2p.dataset import BaseScriptsBuilder


# class ScriptsBuilder(BaseScriptsBuilder):
#     def _get_unique_columns(self) -> List[str]:
#         return ["user_name", "current_url_slug"]

#     def _parse_unique_columns_dict(self, pattern_str: str) -> OrderedDict:
#         user_name, current_url_slug = pattern_str.split("_")

#         result = OrderedDict()
#         result["user_name"] = user_name
#         result["current_url_slug"] = current_url_slug

#         return result


def table_to_records(df: pd.DataFrame, extra_attributes: dict, id_start=0):
    df = df.copy(deep=True)

    df["cells"] = df.apply(
        lambda x: {
            "type": x["cell_type"],
            "content": x["cell_content"].split("\n") if x["cell_content"] else [],
        },
        axis=1,
    )

    df = df.drop(columns=["cell_id", "cell_type", "cell_content"])

    df = (
        df.groupby(["notebook_id"])
        .agg({"user_name": "first", "current_url_slug": "first", "cells": list})
        .reset_index()
    )
    df["notebook_id"] += id_start
    # df = df.drop(columns=['notebook_id'])

    for k, v in extra_attributes.items():
        df[k] = v

    return df
