import json
import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from nb2p import astparse
from nb2p.notebook import Notebook
from webcommon import util
from webcommon.code_repr.ux import UniXcoder


def pad_data(A, size):
    t = size - A.shape[0]
    if len(A.shape) == 2:
        pads = ((0, t), (0, 0))
    else:
        pads = (0, t)
    return np.pad(A, pad_width=pads, mode="constant")


class NotebookLoader:
    def __init__(self, path: str) -> None:
        self._path = path

    def load(self) -> Optional[pd.DataFrame]:
        with open(self._path, "r") as f:
            try:
                notebook = json.loads(f.read())
            except json.JSONDecodeError as e:
                logging.warning(f"Cannot parse JSON")
                return None

        try:
            iter(notebook)
        except TypeError as te:
            logging.warning(f"Cannot iterate over notebook")
            return None

        if (not notebook) or (not "cells" in notebook) or (not notebook["cells"]):
            logging.warning(f"No cell")
            return None

        df = pd.DataFrame(notebook["cells"])
        if ("source" not in df.columns) or ("cell_type" not in df.columns):
            logging.warning(f"No cell type or cell content")
            return None

        df["source"] = df["source"].apply(
            lambda x: "".join(x) if isinstance(x, list) else x
        )

        df["cell_content"] = df["source"]
        df = df[df["cell_content"] != ""]
        df.drop(columns=["source"], inplace=True)

        df.reset_index(names=["cell_id"], inplace=True)
        df = df[
            [
                "cell_id",
                "cell_type",
                "cell_content",
            ]
        ]

        return df

    @property
    def code_lines(self) -> List[str] | None:
        df = self.load()
        if df is None:
            return None

        code = list(df[df["cell_type"] == "code"]["cell_content"].values)
        code: List[str] = "\n".join(code).split("\n")

        return code

    # @property
    # def cells(self) -> List[List[str]]:
    #     df = self.load()
    #     if df is None:
    #         return None

    #     cells = df[df["cell_type"] == "code"]["cell_content"].values.tolist()
    #     cells = [c.split("\n") for c in cells]

    #     return cells


from tree_sitter import Language, Parser


def index_to_code_token(index, code_lines):
    start_point = index[0]
    end_point = index[1]
    if start_point[0] == end_point[0]:
        s = code_lines[start_point[0]][start_point[1] : end_point[1]]
    else:
        s = ""
        s += code_lines[start_point[0]][start_point[1] :]
        for i in range(start_point[0] + 1, end_point[0]):
            s += "\n" + code_lines[i]
        s += "\n" + code_lines[end_point[0]][: end_point[1]]
    return s


class NotebookEmbeddingAST:
    def __init__(
        self,
        notebook: Notebook,
        model: UniXcoder,
        device: torch.device,
        batch_size=50,
    ) -> None:
        self._notebook = notebook
        self._token_ids = []
        self._model = model
        self._device = device
        self._batch_size = batch_size

        PY_LANGUAGE = Language("./build/tree-sitter-python.so", "python")
        parser = Parser()
        parser.set_language(PY_LANGUAGE)
        self._parser = parser

    def get_segment_embeddings(self):
        embeddings = []

        self._token_ids, ast_cl_map = self.tokens_ids()
        self._token_ids = np.array(self._token_ids)
        # pprint(self._token_ids.shape)

        for ti in np.vsplit(
            np.array(self._token_ids),
            np.arange(self._batch_size, len(self._token_ids), self._batch_size),
        ):
            input = torch.tensor(ti).to(self._device)
            # print(input.shape)

            with torch.no_grad():
                tokens_embeddings, code_embedding = self._model(input)
            result = code_embedding.detach().cpu().numpy()  # type:ignore
            embeddings.append(result.squeeze())

            del input
            del tokens_embeddings
            del code_embedding

        return embeddings, ast_cl_map

    def get(self):
        segment_embeddings, ast_cl_map = self.get_segment_embeddings()
        # ground_truth = np.array(codeop.segment_ends_to_binary(segment_ends))

        return np.vstack(segment_embeddings), ast_cl_map

    def _get_ast_children_code(self, code_lines: List[str]):
        code = "\n".join(code_lines)
        tree = self._parser.parse(code.encode())
        root = tree.root_node

        error_lines = set()

        for child in root.children:
            if child.start_point[0] in error_lines or child.type == "ERROR":
                error_lines.add(child.start_point[0])
                continue

            yield index_to_code_token(
                (child.start_point, child.end_point), code_lines
            ), (child.end_point[0] - child.start_point[0] + 1)

    def tokens_ids(self):
        MAX_NUM_TOKENS = 1024
        config = self._model.config
        tokenizer = self._model.tokenizer

        result = []
        segment_ends = []
        ast_cl_map = []

        for s in self._notebook.segments:
            code_lines = s.code_lines

            children_code, a_ast_cl_map = list(
                map(list, zip(*self._get_ast_children_code(code_lines)))
            )
            print(children_code, a_ast_cl_map)
            print()

            ast_cl_map.extend(a_ast_cl_map)

            for code in children_code:
                # print(code)
                # print()
                ast = astparse.AST(code, "python", tokenizer)
                # print(f"len(ast): {len(ast)}")

                tokens = (
                    [tokenizer.cls_token]
                    + ["<encoder-only>"]
                    + [tokenizer.sep_token]
                    + ast
                    + [tokenizer.eos_token]
                )
                tokens = tokens[: MAX_NUM_TOKENS - 4]
                tokens_id = tokenizer.convert_tokens_to_ids(tokens)

                tokens_id = tokens_id + [config.pad_token_id] * (
                    MAX_NUM_TOKENS - len(tokens_id)
                )
                # print(len(tokens_id))

                result.append(tokens_id)

            segment_ends.append(len(children_code))

        # segment_ends = list(map(lambda x: x - 1, np.cumsum(segment_ends)))

        return result, ast_cl_map


class CodeEmbeddingBuilder:
    def __init__(
        self,
        code: List[str],
        model: UniXcoder,
        device: torch.device,
        segment_length: int = 500,
        batch_size=200,
    ) -> None:
        self._code = code
        self._token_ids = []
        self._model = model
        self._device = device
        self._segment_length = segment_length
        self._batch_size = batch_size

    def build(self) -> np.ndarray:
        return np.vstack(self._get_code_line_embeddings())

    def _get_code_line_embeddings(self):
        embeddings = []

        # print("\n".join(self._code))

        self._token_ids = self._get_token_ids()
        # self._token_ids = list(itertools.chain.from_iterable(self._token_ids))
        # self._token_ids = np.array(self._token_ids)
        # pprint(self._token_ids.shape)

        for ti in np.vsplit(
            np.array(self._token_ids),
            np.arange(self._batch_size, len(self._token_ids), self._batch_size),
        ):
            input = torch.tensor(ti).to(self._device)
            logging.debug(f"input.shape: {input.shape}")

            with torch.no_grad():
                tokens_embeddings, code_embedding = self._model(input)

            result = code_embedding.detach().cpu().numpy()  # type:ignore
            embeddings.append(result.squeeze())

            del input
            del tokens_embeddings
            del code_embedding

        return embeddings

    def _get_token_ids(self):
        result = []

        for s in self._code:
            result.append(
                self._model.tokenize(
                    [s],
                    max_length=512,
                    mode="<encoder-only>",
                    padding=True,
                )[0]
            )

        return result


class CRModel:
    def __init__(self, config: dict) -> None:
        self._config = config

    def load_model(self):
        model = UniXcoder("microsoft/unixcoder-base").to(self.device)
        model.eval()
        self._model = model

    @property
    def device(self) -> torch.device:
        if "device" in self._config:
            device_str: str = self._config["device"]
        else:
            device_str = "cpu"
        
        return util.parse_torch_device(device_str)

    def get_code_lines(self, notebook_path: str) -> Optional[List[str]]:
        notebook = NotebookLoader(notebook_path)

        return notebook.code_lines

    # def get_cells(self, notebook_path: str) -> Optional[List[List[str]]]:
    #     notebook = NotebookLoader(notebook_path)

    #     return notebook.cells

    def get_embedding(
        self,
        notebook_path: Optional[str] = None,
        code_lines: Optional[List[str]] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[List[int]]]:
        if code_lines:
            notebook = Notebook.from_code_strings(code_lines)
        elif notebook_path:
            notebook_df = NotebookLoader(notebook_path).load()
            if not notebook_df:
                return None, None
            notebook = Notebook.from_dataframe(notebook_df)
        else:
            return None, None

        return NotebookEmbeddingAST(notebook, self._model, self.device).get()

        # if len(code) > 186:
        #     return None

        # return pad_data(
        #     CodeEmbeddingBuilder(code, self._model, self.device).build(), 186
        # )
