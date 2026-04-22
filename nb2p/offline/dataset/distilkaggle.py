
from turtle import pd


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
