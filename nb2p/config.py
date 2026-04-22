"""Configuration constructor."""

import os
from pathlib import Path
from typing import Any, List

import toml
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.


class DirectoryConfig:
    def __init__(self, base_dir: str) -> None:
        self.base = Path(base_dir)  # Base dir

    @property
    def notebook(self) -> Path:  # Notebook dir
        return self.base / "ipynb"

    @property
    def cr(self) -> Path:  # Code Representation
        return self.base / "processed-unixcoder-full"

    @property
    def astcr(self) -> Path:  # Code Representation (AST)
        return self.base / "processed-unixcoder-ast"
    
    @property
    def dfgtree(self) -> Path:
        return self.base / "dfgtree"

    @property
    def edacr(self) -> Path:  # Code Representation (EDA Notebooks)
        return self.base / "processed-unixcoder-eda"

    @property
    def model(self) -> Path:  # Trained models
        return self.base / "models" / "full"

    @property
    def log(self) -> Path:  # Train log
        return self.base / "logs" / "full"

    def makedirs(self) -> None:
        for attr in [
            a
            for a in dir(self)
            if not a.startswith("_") and type(getattr(self, a, "")).__name__ != "method"
        ]:
            print(f"making dirs: {self.__getattribute__(attr)}")
            os.makedirs(self.__getattribute__(attr), exist_ok=True)


def load_config(config_file_path: str):
    return toml.load(config_file_path)


def parse_path(path_str: str):
    """Parse path string.

    The parsing considers both relative and absolute paths. If a path is
    relative, concat it with $NB2P_PREFIX.
    """
    if os.path.isabs(path_str):
        return path_str
    else:
        return os.path.abspath(os.path.join(os.environ["NB2P_PREFIX"], path_str))


default_config = load_config(parse_path("config.toml"))


def value(key_str: str = "", keys: List[str] = []):
    """Get config value using dot (.) separated key string."""

    value = default_config

    if key_str:
        keys = key_str.split(".")

    for k in keys:
        value: Any = value[k]

    return value


def dirs(dataset_name: str) -> DirectoryConfig:
    return DirectoryConfig(
        parse_path(value(keys=["nbsegment", dataset_name, "prefix"]))
    )
