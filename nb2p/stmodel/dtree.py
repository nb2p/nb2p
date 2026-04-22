import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

from sklearn.tree import DecisionTreeClassifier

from nb2p import codeop
from nb2p.config import DirectoryConfig

from . import ModelInput
from .rforest import RForestModel
from .sklearn import SklearnModel


class DTreeModel(SklearnModel):
    def __init__(
        self,
        params: dict,
        log_path: Union[str, bytes, os.PathLike] = ".",
        model_path: Union[str, bytes, os.PathLike] = "",
    ) -> None:
        super().__init__(
            DecisionTreeClassifier,
            params=params,
            log_path=log_path,
            model_path=model_path,
        )


@dataclass
class DTreeConfig:
    max_depth: int
    min_samples_split: float = 0.001
    n_jobs: int = 80


def train_test(
    train_data: Optional[Tuple[ModelInput, ModelInput]],
    test_data: Tuple[ModelInput, ModelInput],
    test_lengths: list,
    code: list,
    log_path: Union[str, bytes, os.PathLike],
    model_path: Union[str, bytes, os.PathLike],
    config: DTreeConfig,
    trained=True,
):
    model = RForestModel(
        params={
            "random_state": 42,
            "n_estimators": 1,
            "n_jobs": config.n_jobs,
            "verbose": 10,
            "max_depth": config.max_depth,
            "min_samples_split": config.min_samples_split,
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


def default_log_path(dirs: DirectoryConfig, setup_name: str, config: DTreeConfig):
    return (
        dirs.log
        / f"{setup_name}-dtree-depth_{config.max_depth}-minsplit_{config.min_samples_split}"
    )


def default_model_path(dirs: DirectoryConfig, setup_name: str, config: DTreeConfig):
    return (
        dirs.model
        / f"{setup_name}-dtree-depth_{config.max_depth}-minsplit_{config.min_samples_split}.joblib"
    )
