import json
import os
import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple, Type, Union

import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier

from nb2p import fileop, npop, codeop
from nb2p.config import DirectoryConfig

from . import BaseModel, ModelInput
from .sklearn import SklearnModel


class XGBoostModel(SklearnModel):
    def __init__(
        self,
        params: dict,
        log_path: Union[str, bytes, os.PathLike] = ".",
        model_path: Union[str, bytes, os.PathLike] = "",
    ) -> None:
        super().__init__(
            XGBClassifier,
            params=params,
            log_path=log_path,
            model_path=model_path,
        )


class XGBoostModelCPU(BaseModel):
    def __init__(
        self,
        params: dict,
        log_path: Union[str, bytes, os.PathLike] = ".",
        model_path: Union[str, bytes, os.PathLike] = "",
    ) -> None:
        super().__init__(log_path=log_path, params=params)
        self._params = dict()
        self._params.update(params)
        self._model = None
        self._booster = None
        self._model_path = f"./xgboostcpu" if model_path == "" else model_path

    @property
    def model(self):
        # return self._model
        return self._booster

    # @model.setter
    # def model(self, value):
    #     # self._model = value
    #     return self._booster

    def load(self, read_path: Optional[Union[str, bytes, os.PathLike]] = None):
        if read_path:
            self._model_path = read_path

        # self._model = fileop.read_joblib(self._model_path)
        self._booster = xgb.Booster()
        self._booster.load_model(str(self._model_path))

        print(f"[{json.dumps(self._params)}] model loaded from: {self._model_path}")

    def save(self, write_path: Optional[Union[str, bytes, os.PathLike]] = None):
        if write_path:
            self._model_path = write_path

        # fileop.write_joblib(self._model, self._model_path)
        if not self._booster:
            raise ValueError("Model not loaded")

        self._booster.save_model(str(self._model_path))
        print(f"[{json.dumps(self._params)}] model dumped to: {self._model_path}")

    def train(self, X: ModelInput, y: ModelInput):
        start_time = time.time()
        self._model = xgb.QuantileDMatrix(
            np.array(npop.to_1d(X)), np.array(npop.to_1d(y))  # type:ignore
        )
        # self._model.fit(npop.to_1d(X), npop.to_1d(y))
        self._booster = xgb.train(self._params, self._model)
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
        if not self._booster:
            raise ValueError("Model not loaded")

        start_time = time.time()
        y_predict = self._booster.predict(
            xgb.QuantileDMatrix(np.array(npop.to_1d(X)))  # type:ignore
        )
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

    def inference_proba(self, X: ModelInput):
        if not self._booster:
            raise ValueError("Model not loaded")

        start_time = time.time()
        y_predict = self._booster.predict(
            xgb.QuantileDMatrix(np.array(npop.to_1d(X)))  # type:ignore
        )
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


class XGBoostModelGPU(BaseModel):
    def __init__(
        self,
        params: dict,
        log_path: Union[str, bytes, os.PathLike] = ".",
        model_path: Union[str, bytes, os.PathLike] = "",
        subsample: float = 1.0,
    ) -> None:
        super().__init__(log_path=log_path, params=params)
        self._params = dict()
        self._params["device"] = "cuda"
        self._params["subsample"] = subsample
        self._params["sampling_method"] = "gradient_based"
        self._params.update(params)
        self._model = None
        self._model_path = f"./xgboostgpu" if model_path == "" else model_path

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

    def train(self, X: Sequence[Any], y: Sequence[Any]):
        start_time = time.time()
        self._model = xgb.QuantileDMatrix(
            np.array(npop.to_1d(X)), np.array(npop.to_1d(y))
        )
        # self._model.fit(npop.to_1d(X), npop.to_1d(y))
        xgb.train(self._params, self._model)
        end_time = time.time()

        with open(self._log_path, "a") as f:
            print(
                (
                    f"[{json.dumps(self._params)}] train time on {len(X)} samples:"
                    f" {end_time - start_time}"
                ),
                file=f,
            )

    def inference(self, X: Sequence[Any], binarize: bool = False):
        start_time = time.time()
        y_predict = self._model.predict(np.array(npop.to_1d(X)))  # type:ignore
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
        y_predict = self._model.predict_proba(npop.to_1d(X))  # type:ignore
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


@dataclass
class XGBoostConfig:
    max_depth: int
    n_estimators: int = 100
    min_samples_split: float = 0.001


def train_test(
    train_data: Tuple[ModelInput, ModelInput],
    test_data: Tuple[ModelInput, ModelInput],
    test_lengths: list,
    code: list,
    log_path: Union[str, bytes, os.PathLike],
    model_path: Union[str, bytes, os.PathLike],
    config: XGBoostConfig,
    trained=True,
):
    if train_data:
        n_train_samples = len(train_data[0])

    model = XGBoostModelCPU(
        params={
            "random_state": 42,
            "num_boost_round": config.n_estimators,
            "n_jobs": config.n_estimators,
            "verbosity": 3,
            "max_depth": config.max_depth,
            "min_child_weight": (
                int(config.min_samples_split * n_train_samples) if train_data else None
            ),
        },
        log_path=log_path,
        model_path=model_path,
    )

    if not trained:
        if not train_data:
            raise ValueError("train data not provided")
        model.train(train_data[0], train_data[1])
        model.save()
    else:
        model.load()

    y_predict = model.inference(test_data[0], binarize=True)
    model.compute_f1_metrics(test_data[1], y_predict, test_lengths, file=True)

    TEST_Y = test_data[1]
    segment_result = [
        {
            "code": code[i],
            "pred_ast": codeop.binary_to_segment_ends(y_predict[i]),
            "gt_ast": codeop.binary_to_segment_ends(TEST_Y[i]),
        }
        for i in range(len(code))
    ]

    return model, segment_result


def default_log_path(dirs: DirectoryConfig, setup_name: str, config: XGBoostConfig):
    return (
        dirs.log
        / f"{setup_name}-xgboost-depth_{config.max_depth}-minsplit_{config.min_samples_split}"
    )


def default_model_path(dirs: DirectoryConfig, setup_name: str, config: XGBoostConfig):
    return (
        dirs.model
        / f"{setup_name}-xgboost-depth_{config.max_depth}-minsplit_{config.min_samples_split}.joblib"
    )
