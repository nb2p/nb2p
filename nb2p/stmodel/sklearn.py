import json
import os
import time
from abc import abstractmethod
from typing import Any, Optional, Sequence, Type, Union

from nb2p import fileop, npop

from . import BaseModel, ModelInput


class SklearnModel(BaseModel):
    def __init__(
        self,
        model_cls: Type[Any],
        params: dict,
        log_path: Union[str, bytes, os.PathLike] = ".",
        model_path: Union[str, bytes, os.PathLike] = "",
    ) -> None:
        super().__init__(log_path=log_path, params=params)
        self._model: Any = model_cls(**params)
        self._model_path = f"./{model_cls.__name__}" if model_path == "" else model_path

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    def load(self, read_path: Optional[Union[str, bytes, os.PathLike]] = None):
        if read_path:
            self._model_path = read_path

        self._model = fileop.read_joblib(self._model_path)
        print(f"[{json.dumps(self._params)}] model loaded from: {self._model_path}")

    def save(self, write_path: Optional[Union[str, bytes, os.PathLike]] = None):
        if write_path:
            self._model_path = write_path

        fileop.write_joblib(self._model, self._model_path)
        print(f"[{json.dumps(self._params)}] model dumped to: {self._model_path}")

    def train(self, X: ModelInput, y: ModelInput):
        start_time = time.time()
        self._model.fit(npop.to_1d(X), npop.to_1d(y))
        end_time = time.time()

        with open(self._log_path, "a") as f:
            print(
                (
                    f"[{json.dumps(self._params)}] train time on {len(X)} samples:"
                    f" {end_time - start_time}"
                ),
                file=f,
            )

    def inference(self, X: ModelInput, binarize: bool = False):
        start_time = time.time()
        y_predict = self._model.predict(npop.to_1d(X))
        end_time = time.time()

        with open(self._log_path, "a") as f:
            print(
                (
                    f"[{json.dumps(self._params)}] inference time on {len(X)} samples:"
                    f" {end_time - start_time}"
                ),
                file=f,
            )

        if binarize:
            y_predict = npop.multi_label_binary(y_predict)

        return y_predict

    def inference_proba(self, X: Sequence[Any]):
        start_time = time.time()
        y_predict = self._model.predict_proba(npop.to_1d(X))
        end_time = time.time()

        with open(self._log_path, "a") as f:
            print(
                (
                    f"[{json.dumps(self._params)}] inference_proba time on"
                    f" {len(X)} samples: {end_time - start_time}"
                ),
                file=f,
            )

        return y_predict
