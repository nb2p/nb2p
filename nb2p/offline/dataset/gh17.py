from collections import OrderedDict
from typing import List

import pandas as pd

from nb2p.dataset import BaseScriptsBuilder


class ScriptsBuilder(BaseScriptsBuilder):
    def _get_unique_columns(self) -> List[str]:
        return ["notebook_id"]

    def _parse_unique_columns_dict(self, pattern_str: str) -> OrderedDict:
        _, notebook_id = pattern_str.split("_")

        result = OrderedDict()
        result["notebook_id"] = int(notebook_id)

        return result


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

    df = df.groupby(["notebook_id"]).agg({"cells": list}).reset_index()
    df["notebook_id"] += id_start
    # df = df.drop(columns=['notebook_id'])

    for k, v in extra_attributes.items():
        df[k] = v

    return df
