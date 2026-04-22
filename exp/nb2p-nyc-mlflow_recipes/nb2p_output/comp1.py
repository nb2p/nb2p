DATA_LOCATION = "../data/nyc.parquet"
df = pd.read_parquet(DATA_LOCATION, index_col=0)


def create_dataset_filter(dataset: pd.DataFrame) -> pd.Series[bool]:
    return (
        (dataset["fare_amount"] > 0)
        & (dataset["trip_distance"] < 400)
        & (dataset["trip_distance"] > 0)
        & (dataset["fare_amount"] < 1000)
    ) | (~dataset.isna().any(axis=1))


df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df, test_size=0.25, random_state=42)
df_train, df_val, df_test = (
    create_dataset_filter(df_train),
    create_dataset_filter(df_val),
    create_dataset_filter(df_test),
)
