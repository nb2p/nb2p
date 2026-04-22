import os
import warnings
from abc import ABCMeta, abstractmethod
from io import TextIOWrapper
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning

from nb2p import npop


class BaseModel(metaclass=ABCMeta):
    def __init__(
        self, log_path: Union[str, bytes, os.PathLike] = ".", params: dict = {}
    ) -> None:
        if issubclass(log_path.__class__, os.PathLike):
            self._log_path = log_path
        else:
            self._log_path = Path(log_path)  # type:ignore
        self._params = params

    @abstractmethod
    def load(self, read_path: Union[str, bytes, os.PathLike]):
        raise NotImplementedError()

    @abstractmethod
    def save(self, write_path: Union[str, bytes, os.PathLike]):
        raise NotImplementedError()

    @abstractmethod
    def train(self, X: Sequence[Any], y: Sequence[Any]):
        raise NotImplementedError()

    @abstractmethod
    def inference(self, X: Sequence[Any]):
        raise NotImplementedError()

    @abstractmethod
    def inference_proba(self, X: Sequence[Any]):
        raise NotImplementedError()

    def compute_f1_metrics(
        self,
        true: Union[Sequence[Any], np.ndarray],
        predict: Union[Sequence[Any], np.ndarray],
        lengths: List[int],
        file: Optional[Union[bool, TextIOWrapper]] = None,
    ):
        METRIC_FUNC = npop.compute_all_metrics_v2

        real_file = None
        if isinstance(file, bool):
            if file is True:
                real_file = open(self._log_path, "a")
                with warnings.catch_warnings():
                    METRIC_FUNC(
                        str(self._params), true, predict, lengths, file=real_file
                    )
                real_file.close()
                return
            else:
                real_file = None
        else:
            real_file = file

        METRIC_FUNC(str(self._params), true, predict, lengths, file=real_file)
