from sklearn.linear_model import SGDRegressor

estimator_params = {}
model = SGDRegressor(random_state=42, **estimator_params)
model.fit(df_train.drop(columns=["fare_amount"]), df_train["fare_amount"])
nyc_output = model.predict(df_val.drop(columns=["fare_amount"]))
nyc_output.to_parquet("../data/nyc_output.parquet")
