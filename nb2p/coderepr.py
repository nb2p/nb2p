"""Code representation (notebook embedding)."""

import sys

if sys.version_info >= (3, 11):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

from typing import List

import numpy as np
import torch

from . import astparse, codeop
from .notebook import Notebook
from .reprmodel.ux import UniXcoder


class Segment(TypedDict):
    prompts: List[str]
    code_lines: List[str]


class NbSegments(TypedDict):
    notebook_id: int
    segments: List[Segment]


class NotebookEmbedding:
    def __init__(
        self,
        notebook: Notebook,
        model: UniXcoder,
        batch_size=100,
    ) -> None:
        self._notebook = notebook
        self._token_ids = []
        self._model = model
        self._batch_size = batch_size
        self._parser = astparse.parser()

    def get_segment_embeddings(self):
        embeddings = []

        self._token_ids, segment_ends = self.tokens_ids()
        self._token_ids = np.array(self._token_ids)

        for ti in np.vsplit(
            np.array(self._token_ids),
            np.arange(self._batch_size, len(self._token_ids), self._batch_size),
        ):
            input = torch.tensor(ti).to(self._model.device)

            with torch.no_grad():
                tokens_embeddings, code_embedding = self._model(input)
            result = code_embedding.detach().cpu().numpy()  # type:ignore
            embeddings.append(result.squeeze())

            del input
            del tokens_embeddings
            del code_embedding

        return embeddings, segment_ends

    def get(self):
        segment_embeddings, segment_ends = self.get_segment_embeddings()
        ground_truth = np.array(codeop.segment_ends_to_binary(segment_ends))

        return np.vstack(segment_embeddings), ground_truth

    def _get_ast_children_code(self, code_lines: List[str]):
        code = "\n".join(code_lines)
        tree = self._parser.parse(code.encode())
        root = tree.root_node

        error_lines = set()

        for child in root.children:
            if child.start_point[0] in error_lines or child.type == "ERROR":
                error_lines.add(child.start_point[0])
                continue

            yield astparse.index_to_code_token(
                (child.start_point, child.end_point), code_lines
            )

    def tokens_ids(self):
        MAX_NUM_TOKENS = 1024
        config = self._model.config
        tokenizer = self._model.tokenizer

        result = []
        segment_ends = []

        for s in self._notebook.segments:
            code_lines = s.code_lines

            children_code = list(self._get_ast_children_code(code_lines))

            for code in children_code:
                ast = astparse.AST(code, "python", tokenizer)

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

                result.append(tokens_id)

            segment_ends.append(len(children_code))

        segment_ends = list(map(lambda x: x - 1, np.cumsum(segment_ends)))

        return result, segment_ends
