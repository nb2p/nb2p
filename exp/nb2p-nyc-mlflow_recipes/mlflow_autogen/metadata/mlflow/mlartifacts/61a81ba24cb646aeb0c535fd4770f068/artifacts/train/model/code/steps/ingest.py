import logging

import pandas as pd
import sklearn
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def create_dataset_filter(dataset: pd.DataFrame) -> pd.Series[bool]:
    return (
        (dataset["fare_amount"] > 0)
        & (dataset["trip_distance"] < 400)
        & (dataset["trip_distance"] > 0)
        & (dataset["fare_amount"] < 1000)
    ) | (~dataset.isna().any(axis=1))


def comp1(file_path: str):
    df = pd.read_parquet(file_path, index_col=0)

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    df_train, df_val = train_test_split(df, test_size=0.25, random_state=42)
    df_train, df_val, df_test = (
        create_dataset_filter(df_train),
        create_dataset_filter(df_val),
        create_dataset_filter(df_test),
    )

    OUTPUT = df_train, df_val, df_test
    return OUTPUT
