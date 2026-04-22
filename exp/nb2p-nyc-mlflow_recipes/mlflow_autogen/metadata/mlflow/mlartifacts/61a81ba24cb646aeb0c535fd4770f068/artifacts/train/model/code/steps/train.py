import logging

import pandas as pd
import sklearn
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def comp3():
    estimator_params = {}

    OUTPUT = SGDRegressor(random_state=42, **estimator_params)
    return OUTPUT
