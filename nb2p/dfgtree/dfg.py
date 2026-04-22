import argparse
import logging
import sys
from dataclasses import dataclass
from nb2p.parser import (
    DFG_go,
    DFG_java,
    DFG_javascript,
    DFG_php,
    DFG_python,
    DFG_ruby,
    index_to_code_token,
    remove_comments_and_docstrings,
    tree_to_token_index,
)
from typing import List, Optional, Tuple

import numpy as np
import torch
import tree_sitter_python as tspython
from torch import nn
from transformers import PreTrainedTokenizer
from tree_sitter import Language, Parser

from nb2p import config

# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)

dfg_function = {
    "python": DFG_python,
    "java": DFG_java,
    "ruby": DFG_ruby,
    "go": DFG_go,
    "php": DFG_php,
    "javascript": DFG_javascript,
}


DEVICE = config.value("code_repr.device")


# load parsers
parsers = {}
for lang in ["python"]:
    # LANGUAGE = Language("parser/my-languages.so")
    PY_LANGUAGE = Language(tspython.language())
    parser = Parser(PY_LANGUAGE)
    # parser.set_language(LANGUAGE)
    parser = [parser, dfg_function[lang]]
    parsers[lang] = parser


# remove comments, tokenize code and extract dataflow
def extract_dataflow(code, parser, lang):
    # remove comments
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:
        pass
    # obtain dataflow
    if lang == "php":
        code = "<?php" + code + "?>"
    try:
        tree = parser[0].parse(bytes(code, "utf8"))
        root_node = tree.root_node
        tokens_index = tree_to_token_index(root_node)
        code = code.split("\n")
        code_tokens = [index_to_code_token(x, code) for x in tokens_index]
        index_to_code = {}
        for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx, code)
        try:
            DFG, _ = parser[1](root_node, index_to_code, {})
        except:
            DFG = []
        DFG = sorted(DFG, key=lambda x: x[1])
        indexs = set()
        for d in DFG:
            if len(d[-1]) != 0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG = []
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg = new_DFG
    except:
        dfg = []
    return code_tokens, dfg


@dataclass
class InputFeatures(object):
    """A single training/test features for a example."""

    code_tokens: list
    code_ids: list
    position_idx: list
    dfg_to_code: list
    dfg_to_dfg: list

    def __repr__(self):
        result = ""
        result += "code_tokens ({}): {}\n".format(
            len(self.code_tokens), [x.replace("\u0120", "_") for x in self.code_tokens]
        )
        result += "code_ids ({}): {}\n".format(
            len(self.code_ids), " ".join(map(str, self.code_ids))
        )
        result += "position_idx ({}): {}\n".format(
            len(self.position_idx), " ".join(map(str, self.position_idx))
        )
        result += "dfg_to_code ({}): {}\n".format(
            len(self.dfg_to_code), " ".join(map(str, self.dfg_to_code))
        )
        result += "dfg_to_dfg: ({}) {}\n".format(
            len(self.dfg_to_dfg), " ".join(map(str, self.dfg_to_dfg))
        )

        return result


def log_input(feat: InputFeatures):
    logger.info(
        "code_tokens: {}".format([x.replace("\u0120", "_") for x in feat.code_tokens])
    )
    logger.info("code_ids: {}".format(" ".join(map(str, feat.code_ids))))
    logger.info("position_idx: {}".format(feat.position_idx))
    logger.info("dfg_to_code: {}".format(" ".join(map(str, feat.dfg_to_code))))
    logger.info("dfg_to_dfg: {}".format(" ".join(map(str, feat.dfg_to_dfg))))


def get_features(code_str: str, tokenizer, args):
    max_code = args.code_length
    max_df = args.data_flow_length

    # code
    parser = parsers["python"]

    # extract data flow
    code_tokens, dfg = extract_dataflow(code_str, parser, "python")
    # code_tokens = [
    #     tokenizer.tokenize("@ " + x)[1:] if idx != 0 else tokenizer.tokenize(x)
    #     for idx, x in enumerate(code_tokens)
    # ]
    code_tokens=[tokenizer.encode(x).tokens for idx,x in enumerate(code_tokens)]

    ori2cur_pos = {}
    ori2cur_pos[-1] = (0, 0)
    for i in range(len(code_tokens)):
        ori2cur_pos[i] = (
            ori2cur_pos[i - 1][1],
            ori2cur_pos[i - 1][1] + len(code_tokens[i]),
        )

    code_tokens = [y for x in code_tokens for y in x]

    # truncating
    code_tokens = code_tokens[: max_code + max_df - 2 - min(len(dfg), max_df)]
    code_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    # code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    code_ids =  [tokenizer.token_to_id(tok) for tok in code_tokens]

    position_idx = [i + tokenizer.pad_token_id + 1 for i in range(len(code_tokens))]
    dfg = dfg[: max_code + max_df - len(code_tokens)]
    code_tokens += [x[0] for x in dfg]
    position_idx += [0 for x in dfg]
    code_ids += [tokenizer.unk_token_id for x in dfg]

    padding_length = max_code + max_df - len(code_ids)
    position_idx += [tokenizer.pad_token_id] * padding_length
    code_ids += [tokenizer.pad_token_id] * padding_length

    # reindex
    reverse_index = {}
    for idx, x in enumerate(dfg):
        reverse_index[x[1]] = idx

    for idx, x in enumerate(dfg):
        dfg[idx] = x[:-1] + ([reverse_index[i] for i in x[-1] if i in reverse_index],)

    dfg_to_dfg = [x[-1] for x in dfg]
    dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
    length = len([tokenizer.cls_token])
    dfg_to_code = [(x[0] + length, x[1] + length) for x in dfg_to_code]

    return InputFeatures(
        code_tokens,
        code_ids,
        position_idx,
        dfg_to_code,
        dfg_to_dfg,
    )


def get_attn_mask(
    feat: InputFeatures, code_length: int, data_flow_length: int, device: torch.device
):
    # calculate graph-guided masked function
    attn_mask = np.zeros(
        (
            code_length + data_flow_length,
            code_length + data_flow_length,
        ),
        dtype=bool,
    )

    # calculate begin index of node and max length of input
    node_index = sum([i > 1 for i in feat.position_idx])
    max_length = sum([i != 1 for i in feat.position_idx])

    # sequence can attend to sequence
    attn_mask[:node_index, :node_index] = True

    # special tokens attend to all tokens
    for idx, i in enumerate(feat.code_ids):
        if i in [0, 2]:
            attn_mask[idx, :max_length] = True

    # nodes attend to code tokens that are identified from
    for idx, (a, b) in enumerate(feat.dfg_to_code):
        if a < node_index and b < node_index:
            attn_mask[idx + node_index, a:b] = True
            attn_mask[a:b, idx + node_index] = True

    # nodes attend to adjacent nodes
    for idx, nodes in enumerate(feat.dfg_to_dfg):
        for a in nodes:
            if a + node_index < len(feat.position_idx):
                attn_mask[idx + node_index, a + node_index] = True

    return attn_mask


class CodeEncodingBuilder:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        model: nn.Module,
        device: torch.device = torch.device(DEVICE),
    ) -> None:
        self._tokenizer = tokenizer
    
        tokenizer.cls_token = "<s>"
        tokenizer.sep_token = "</s>"
        tokenizer.pad_token_id = tokenizer.token_to_id("<pad>")
        tokenizer.unk_token_id = tokenizer.token_to_id("<unk>")
    
        if model:
            self._model = model.to(device)

        self._args = argparse.Namespace(code_length=256, data_flow_length=64)
        self._device = device

    def make_input_features(self, code_str: str) -> InputFeatures:
        return get_features(code_str, self._tokenizer, self._args)

    def make_input(
        self,
        code_str: str = "",
        input_features: Optional[InputFeatures] = None,
    ) -> List[torch.Tensor]:
        if not input_features:
            input_features = self.make_input_features(code_str)

        attn_mask = get_attn_mask(
            input_features,
            self._args.code_length,
            self._args.data_flow_length,
            self._device,
        )

        return [
            torch.tensor(input_features.code_ids).to(self._device),
            torch.tensor(attn_mask).to(self._device),
            torch.tensor(input_features.position_idx).to(self._device),
        ]

    def get_encoding(self, input: List[torch.Tensor]):
        # for each input, if the tensor is one dimensional, insert a batch dimension
        if input[0].ndim <= 1:
            input[0] = input[0][None, :]

        if input[1].ndim <= 2:
            input[1] = input[1][None, :]

        if input[2].ndim <= 1:
            input[2] = input[2][None, :]

        return self._model(input[0], input[1], input[2])

    def __repr__(self) -> str:
        return f"<CodeEncodingBuilder device={self._device}>"


def get_code_encoding(code_str: str, builder: CodeEncodingBuilder):
    feats = builder.make_input_features(code_str)
    print(feats)
    inp = builder.make_input(input_features=feats)
    out = builder.get_encoding(inp)

    return out
