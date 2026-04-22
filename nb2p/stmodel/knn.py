import os
from typing import Any, Type, Union

from sklearn.neighbors import KNeighborsRegressor

from .sklearn import SklearnModel


class KNNModel(SklearnModel):
    def __init__(
        self,
        params: dict,
        log_path: Union[str, bytes, os.PathLike] = ".",
        model_path: Union[str, bytes, os.PathLike] = "",
    ) -> None:
        super().__init__(
            KNeighborsRegressor,
            params=params,
            log_path=log_path,
            model_path=model_path,
        )
