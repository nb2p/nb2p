import logging

import pandas as pd
import sklearn
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def calculate_features(df: DataFrame):
    df["pickup_dow"] = df["tpep_pickup_datetime"].dt.dayofweek
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    trip_duration = df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    df["trip_duration"] = trip_duration.map(lambda x: x.total_seconds() / 60)
    dateTimeColumns = list(df.select_dtypes(include=["datetime64"]).columns)
    df[dateTimeColumns] = df[dateTimeColumns].astype(str)
    df.drop(columns=["tpep_pickup_datetime", "tpep_dropoff_datetime"], inplace=True)
    return df


def comp2():
    function_transformer_params = (
        {}
        if sklearn.__version__.startswith("1.0")
        else {"feature_names_out": "one-to-one"}
    )

    OUTPUT = Pipeline(
        steps=[
            (
                "calculate_time_and_duration_features",
                FunctionTransformer(calculate_features, **function_transformer_params),
            ),
            (
                "encoder",
                ColumnTransformer(
                    transformers=[
                        (
                            "hour_encoder",
                            OneHotEncoder(categories="auto", sparse=False),
                            ["pickup_hour"],
                        ),
                        (
                            "day_encoder",
                            OneHotEncoder(categories="auto", sparse=False),
                            ["pickup_dow"],
                        ),
                        (
                            "std_scaler",
                            StandardScaler(),
                            ["trip_distance", "trip_duration"],
                        ),
                    ]
                ),
            ),
        ]
    )
    return OUTPUT
