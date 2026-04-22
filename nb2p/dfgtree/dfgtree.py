from functools import cache, cached_property
from itertools import chain
import os
from pprint import pprint
from typing import Dict, List, NamedTuple, Optional, Tuple, TypedDict

import torch

from nb2p import database
import numpy as np
from nb2p.dfgtree.dfg import CodeEncodingBuilder
from tree_sitter import Language, Node, Parser, Tree

# Load model directly
from tokenizers import Tokenizer
from transformers import (
    RobertaTokenizer,
    RobertaModel,
    RobertaConfig,
    RobertaForSequenceClassification,
)

# from model import Model
from nb2p.dfgtree.compressor.model import Model

from nb2p.astparse import parser
from nb2p.notebook import Notebook


def node_to_code(node: Node, code_str: str):
    return code_str[node.start_byte : node.end_byte]


_DEBUG_CODE_STR = """i = 1
from os import (
    path,
    environ
)
import time
def foo():
    if bar:
        baz()
        def qux():
            quux()
def qux(one: str, two, three):
    quux()
qux()"""


def _debug_node_to_code():
    p, lang = parser()

    tree = p.parse(bytes(_DEBUG_CODE_STR, "utf8"))
    print(node_to_code(tree.root_node.children[0], _DEBUG_CODE_STR))


def get_params(params_node: Node, code_str: str):
    if len(params_node.children) == 0:
        return []

    result = []
    for p in params_node.children:
        if p.type == "identifier":
            result.append(node_to_code(p, code_str))
        elif p.type == "typed_parameter":
            id_node = p.children[0]
            assert id_node
            result.append(node_to_code(id_node, code_str))

    return result


def remaining_str(s: str, indices: List[Tuple[int, int]]):
    """Returns a copy of s with the given indices removed."""
    if len(indices) == 0:
        return s

    """flatten the indicies"""
    ids = list(chain([0], list(chain.from_iterable(indices)), [len(s)]))
    assert len(ids) % 2 == 0

    """extract odd and even indices into two lists"""
    id_ranges = list(zip(ids[0::2], ids[1::2]))
    # print(id_ranges)

    """get the substrings"""
    substrings = [s[i:j] for i, j in id_ranges]
    # print(substrings)

    return "".join(substrings)


def _debug_remaining_str():
    print(remaining_str("123456", [(1, 2), (3, 5)]))
    print(remaining_str("123456", [(2, 3), (4, 5)]))


class GlobalFunctionExtractor:
    QUERY_STR = """
(module
    (function_definition
        name: (identifier) @name_node
        parameters: (parameters) @params_node
        body: (block) @block_node
    ) @func_node
)
"""

    def __init__(self, code_str: str, tree: Tree, lang: Language):
        self._code_str = code_str
        self._query = lang.query(self.QUERY_STR)
        self._tree = tree

    @cache
    def get(self):
        matches = self._query.matches(self._tree.root_node)
        result = []

        for m in matches:
            m = m[1]
            func_node = m["func_node"][0]
            name_node = m["name_node"][0]
            block_node = m["block_node"][0]
            params_node = m["params_node"][0]
            params = get_params(params_node, self._code_str)

            result.append(
                {
                    "func_node": func_node,
                    "params": params,
                    "name": node_to_code(name_node, self._code_str),
                    "name_node": name_node,
                    "block_code": node_to_code(block_node, self._code_str),
                    "block_node": block_node,
                }
            )

        return result

    @cached_property
    def remaining_code(self):
        fds = self.get()

        return remaining_str(
            self._code_str,
            [
                # end_byte + 1 to exclude trailing "\n"
                (f["func_node"].start_byte, f["func_node"].end_byte + 1)
                for f in fds
            ],
        )


def get_imports_and_from_imports(params_node: Node, code_str: str):
    if len(params_node.children) == 0:
        return []

    result = []
    for p in params_node.children:
        if p.type == "import_statement":
            id_node = p.children[0]
            assert id_node
            result.append(node_to_code(id_node, code_str))

    return result


class GlobalImportExtractor:
    QUERY_STR = """
(module
    [
        (import_statement) @import_node
        (import_from_statement) @import_node
    ]
)
"""

    def __init__(self, code_str: str, tree: Tree, lang: Language):
        self._code_str = code_str
        self._query = lang.query(self.QUERY_STR)
        self._tree = tree

    @cache
    def get(self):
        matches = self._query.matches(self._tree.root_node)
        result = []

        for m in matches:
            m = m[1]
            import_node = m["import_node"][0]

            result.append(
                {
                    "import_node": import_node,
                    "import_code": node_to_code(import_node, self._code_str),
                }
            )

        return result

    @cached_property
    def remaining_code(self):
        ids = self.get()
        
        # print(self._code_str)

        result = remaining_str(
            self._code_str,
            [
                # end_byte + 1 to exclude trailing "\n"
                (i["import_node"].start_byte, i["import_node"].end_byte + 1)
                for i in ids
            ],
        )
        
        return result

    @property
    def rearranged_code(self):
        """move imports to the top"""

        imports = self.get()
        remaining = self.remaining_code

        return "".join([i["import_code"] + "\n" for i in imports]) + remaining


def _debug_GlobalImportExtractor(code_str, parser, lang):
    tree = parser.parse(bytes(code_str, "utf8"))
    gie = GlobalImportExtractor(code_str, tree, lang)
    print(gie.get())
    print(gie.rearranged_code)

    return gie


def _debug_GlobalFunctionExtractor(code_str, parser, lang):
    tree = parser.parse(bytes(code_str, "utf8"))
    gfe = GlobalFunctionExtractor(code_str, tree, lang)
    print(gfe.get())
    print(gfe.remaining_code)

    return gfe


def _debug_GFE_GIE(parser, lang):
    gie = _debug_GlobalImportExtractor(_DEBUG_CODE_STR, parser, lang)
    _debug_GlobalFunctionExtractor(gie.rearranged_code, parser, lang)


def get_global_children_code_strings(root: Node, code_str: str):
    """
    get the code strings of all nodes that are direct children of root
    node
    """
    result = []

    if len(root.children) == 1:
        if root.children[0].type in [
            "function_definition",
            "class_definition",
            "for_statement",
            "while_statement",
        ]:
            node = root.children[0].child_by_field_name("body")
            assert node
            root = node

    for c in root.children:
        result.append(node_to_code(c, code_str))

    return result


class PreprocessResult(TypedDict):
    imports: List[str]
    func_defs: List[Tuple[str, str]]
    segments: List[str]


def preprocess(nb: Notebook, parser: Parser, lang: Language) -> PreprocessResult:
    result: PreprocessResult = {"imports": [], "func_defs": [], "segments": []}

    for s in nb.segments:
        code_str = "\n".join(s.code_strings)
        t = parser.parse(bytes(code_str, "utf8"))

        imps = []
        gie = GlobalImportExtractor(code_str, t, lang)
        for imp in gie.get():
            imp_code = node_to_code(imp["import_node"], code_str)
            imps.append(imp_code)

        result["imports"].extend(imps)

        code_str = gie.remaining_code
        t = parser.parse(bytes(code_str, "utf8"))
        gfe = GlobalFunctionExtractor(code_str, t, lang)

        fdefs = []
        for d in gfe.get():
            func_code = node_to_code(d["func_node"], code_str)
            fdefs.append((d["name"], func_code))

        result["func_defs"].extend(fdefs)

        code_str = gfe.remaining_code
        t = parser.parse(bytes(code_str, "utf8"))
        result["segments"].append(code_str)

    return result


class DFGNode:
    def __init__(
        self, node_repr: Optional[np.ndarray], children_reprs: List["DFGNode"]
    ):
        self._node_repr = node_repr
        self._children_reprs = children_reprs

    @property
    def repr(self):
        return self._node_repr

    @property
    def children(self):
        return self._children_reprs

    def __repr__(self):
        return (
            (
                f"<DFGNode repr={self._node_repr.shape} children={self._children_reprs} ({len(self._children_reprs)}))>"
            )
            if self._node_repr is not None
            else f"<DFGNode repr=None children={self._children_reprs} ({len(self._children_reprs)})>"
        )


def get_num_ast_children(tree: Tree):
    cursor = tree.walk()
    cursor.goto_first_child()

    result = 1
    while cursor.goto_next_sibling():
        result += 1

    return result


def get_dfg_tree_node(
    depth: int,
    max_depth: int,
    code_str: str,
    builder: CodeEncodingBuilder,
    parser: Parser,
    with_encodings: bool = True,
) -> Tuple[Optional[DFGNode], Optional[int]]:
    is_root_node = depth == 1

    t = parser.parse(bytes(code_str, "utf8"))

    children_reprs = []
    try:
        feats = builder.make_input_features(code_str)
    except:
        return None, None

    out = None
    if (not is_root_node) and with_encodings:
        inp = builder.make_input(input_features=feats)
        out = builder.get_encoding(inp).cpu().detach().numpy()

    is_not_too_deep = depth < max_depth
    is_truncated = feats and len(feats.code_tokens) == 320

    gnodes = []
    if (is_root_node) or (is_not_too_deep and is_truncated):
        """The code has been truncated"""
        gnodes = get_global_children_code_strings(t.root_node, code_str)

        for n in gnodes:
            child_node, _ = get_dfg_tree_node(
                depth + 1, max_depth, n, builder, parser, with_encodings=with_encodings
            )
            if child_node is None:
                continue

            children_reprs.append(child_node)

    return DFGNode(out, children_reprs), len(gnodes)


class DFGTree:
    def __init__(self, input: PreprocessResult, builder: CodeEncodingBuilder):
        self._input = input
        self._builder = builder

    def build(self, with_encodings: bool = True) -> Dict:
        p, lang = parser()

        result = {"func_defs": [], "segments": [], "segment_ends": [-1]}

        for f in self._input["func_defs"]:
            r, n = get_dfg_tree_node(
                1, 4, f[1], self._builder, p, with_encodings=with_encodings
            )

            if with_encodings:
                result["func_defs"].append(
                    {"name": f[0], "code": f[1], "repr": r, "n_ast_children": n}
                )
            else:
                result["func_defs"].append(
                    {"name": f[0], "code": f[1], "n_ast_children": n}
                )

        for s in self._input["segments"]:
            r, n = get_dfg_tree_node(
                1, 4, s, self._builder, p, with_encodings=with_encodings
            )

            if with_encodings:
                result["segments"].append({"code": s, "repr": r, "n_ast_children": n})
            else:
                result["segments"].append({"code": s, "n_ast_children": n})

            result["segment_ends"].append(result["segment_ends"][-1] + n)

        result["segment_ends"].pop(0)

        return result


def build_dfg_tree(
    notebook: Notebook,
    parser: Parser,
    lang: Language,
    builder: CodeEncodingBuilder,
    with_encodings: bool = True,
):
    preproc_result = preprocess(notebook, parser, lang)
    # print(preproc_result)
    return DFGTree(input=preproc_result, builder=builder).build(
        with_encodings=with_encodings
    )


def get_xs_model(model_dir: str, size: int):
    config = RobertaConfig.from_pretrained("microsoft/graphcodebert-base")
    config.num_attention_heads = 8
    config.hidden_size = 96
    config.intermediate_size = 64
    config.vocab_size = 1000
    config.num_hidden_layers = 12
    config.hidden_dropout_prob = 0.2

    tokenizer_path = os.path.join(
        "compressor", "BPE" + "_" + str(config.vocab_size) + ".json"
    )
    tokenizer = Tokenizer.from_file(tokenizer_path)

    model = Model(RobertaForSequenceClassification(config=config), config, tokenizer)

    model_dir = os.path.join(model_dir, str(size), "model.bin")
    model.load_state_dict(torch.load(model_dir))

    return model, tokenizer


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.get_num_threads()

    DEVICE = torch.device("cuda")

    p, lang = parser()
    db, client = database.connect(dataset_name="distilkaggle", verbose=True)

    model, tokenizer = get_xs_model(
        "compressor/GraphCodeBERT/clone_detection/checkpoint", 3
    )
    model.eval()

    builder = CodeEncodingBuilder(tokenizer, model, DEVICE)

    a_notebook_data = database.get_notebooks(
        db, {"prompted": True, "segments.6": {"$exists": True}}, include_segments=True
    ).next()
    a_notebook = Notebook.from_db_result(a_notebook_data)
    pprint(build_dfg_tree(a_notebook, p, lang, builder, with_encodings=False))

    # _debug_node_to_code()
    # _debug_remaining_str()
    # _debug_GlobalFunctionExtractor()
    # _debug_GlobalImportExtractor()
    # _debug_GFE_GIE(p, lang)
