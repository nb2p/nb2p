import logging
import sys
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, List, Sequence, Tuple

import xgboost as xgb
from loguru import logger

if sys.version_info < (3, 11):
    from typing_extensions import NotRequired, TypedDict
else:
    from typing import NotRequired, TypedDict

import numpy as np
import torch

from nb2p import config, npop
from nb2p.stmodel.dnn import NB2PSS, Transformer
from nb2p.stmodel.rforest import RForestModel
from nb2p.stmodel.xgboost import XGBoostModelCPU
from webcommon import util

DIRS = config.dirs(dataset_name="kgtorrent")
DIRS.makedirs()

MAX_LENGTH = 384
SETUP_NAME = f"8020_astn4_{MAX_LENGTH}"


class Config(TypedDict):
    enabled: NotRequired[bool]
    model_path: NotRequired[str]
    device: NotRequired[str]


class SegmentTransitionModel(metaclass=ABCMeta):
    def __init__(self, config: Config) -> None:
        self._config = config
        self._model: Any = None

    @property
    def enabled(self) -> bool:
        return self._config.get("enabled") == True

    @property
    def device(self) -> torch.device:
        return util.parse_torch_device(self._config.get("device"))

    def load_model(self) -> None:
        model_path = self._config.get("model_path")
        if not model_path:
            raise FileNotFoundError(f"No such model: {model_path}")

        self._model = self._do_load_model(model_path)

    @abstractmethod
    def _do_load_model(self, model_path: str) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def predict_proba(self, code_embeddings: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def _do_inference(
        self,
        code_embeddings: np.ndarray,
        func_do_inference: Callable[[Sequence, Sequence], np.ndarray],
        detach_after_repr: bool,
        use_mask: bool,
    ):
        n_ast_children = code_embeddings.shape[0]

        X = torch.from_numpy(
            np.array(npop.pad(code_embeddings, MAX_LENGTH)[np.newaxis, ...])
        ).to(self.device)

        print(X.shape)

        if detach_after_repr:
            X = X.detach().cpu().numpy()
            X = npop.to_1d(X)

        if use_mask:
            y_mask = npop.mask(n_ast_children, MAX_LENGTH)[np.newaxis, ...]
            y_pred_eval = func_do_inference(X, y_mask)
        else:
            y_pred_eval = func_do_inference(X, [])

        print(y_pred_eval)

        # y_pred_eval = np.array(y_pred_eval)
        y_pred = npop.multi_label_binary(y_pred_eval)
        y_pred = y_pred[0][:n_ast_children]
        y_pred[-1] = 1

        return y_pred

    def predict(self, code_embeddings: np.ndarray) -> np.ndarray:
        return np.where(self.predict_proba(code_embeddings) >= 0.5, 1, 0)

    def segment_ends(
        self, code_embeddings: np.ndarray, ast_cl_map: List[int]
    ) -> Tuple[List[int], List[int]]:
        ast_ends: List[int] = (
            np.asarray(self.predict(code_embeddings) > 0.5).nonzero()[0].tolist()
        )

        logger.debug(f"AST ENDS: {ast_ends}")

        ast_ends.insert(0, 0)

        result = []
        for i in range(1, len(ast_ends)):
            last_end = 0 if len(result) == 0 else result[-1]

            logger.debug("{} {} {}", ast_ends[i - 1], ast_ends[i], last_end)

            new_lines = 0
            for k in range(ast_ends[i - 1] + 1, ast_ends[i] + 1):
                new_lines += ast_cl_map[k]

            result.append(last_end + new_lines)

        return ast_ends[1:], result


class SklearnSTModel(SegmentTransitionModel):
    def predict_proba(self, code_embeddings: np.ndarray) -> np.ndarray:
        return self._do_inference(
            code_embeddings,
            lambda X, _: self._model.inference(X),
            detach_after_repr=True,
            use_mask=False,
        )


class DecisionTreeSTModel(SklearnSTModel):
    max_depth = 64
    min_samples_split = 0.001

    def _do_load_model(self, model_path: str) -> Any:
        model = RForestModel(
            params={
                "random_state": 42,
                "n_estimators": 1,
                "n_jobs": 80,
                "verbose": 10,
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
            },
            model_path=model_path,
            log_path="/dev/null",
        )
        model.load()
        return model


class RandomForestSTModel(SklearnSTModel):
    max_depth = 64
    min_samples_split = 0.001
    n_estimators = 100

    def _do_load_model(self, model_path: str) -> Any:
        model = RForestModel(
            params={
                "random_state": 42,
                "n_estimators": self.n_estimators,
                "n_jobs": self.n_estimators,
                "verbose": 10,
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
            },
            model_path=model_path,
            log_path="/dev/null",
        )
        model.load()
        return model


class XGBoostSTModel(SklearnSTModel):
    max_depth = 8
    n_estimators = 100

    def _do_load_model(self, model_path: str) -> Any:
        model = XGBoostModelCPU(
            params={
                "random_state": 42,
                "num_boost_round": self.n_estimators,
                "n_jobs": self.n_estimators,
                "verbosity": 3,
                "max_depth": self.max_depth,
            },
            model_path=model_path,
            log_path="/dev/null",
        )
        model.load()
        return model


class TransformerSTModel(SegmentTransitionModel):
    num_heads = 8
    hidden_size = 512
    num_layers = 4

    def _do_load_model(self, model_path: str) -> Any:
        model = Transformer(
            MAX_LENGTH,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_heads=1,
        ).to(self.device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model

    def predict_proba(self, code_embeddings: np.ndarray) -> np.ndarray:
        n_ast_children = code_embeddings.shape[0]

        model_input = torch.from_numpy(npop.pad(code_embeddings, MAX_LENGTH)).to(
            self.device
        )
        model_input = torch.unsqueeze(model_input, dim=0)
        logging.debug(f"code_embeddings.shape: {model_input.shape}")

        with torch.no_grad():
            y_mask = torch.tensor(
                npop.mask(n_ast_children, MAX_LENGTH)[np.newaxis, ...]
            ).to(self.device)
            model_output: torch.Tensor = self._model(model_input, y_mask).detach().cpu()

        result = model_output.numpy().squeeze(0)

        return result


class NB2PSSSTModel(SegmentTransitionModel):
    num_heads = 8
    hidden_size = 512
    num_layers = 4

    def _do_load_model(self, model_path: str) -> Any:
        model = NB2PSS(
            MAX_LENGTH,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        ).to(self.device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model

    def predict_proba(self, code_embeddings: np.ndarray) -> np.ndarray:
        model_input = torch.from_numpy(npop.pad(code_embeddings, MAX_LENGTH)).to(
            self.device
        )
        model_input = torch.unsqueeze(model_input, dim=0)
        logging.debug(f"code_embeddings.shape: {model_input.shape}")

        with torch.no_grad():
            model_output: torch.Tensor = self._model(model_input, None).detach().cpu()

        result = model_output.numpy().squeeze(0)

        return result
