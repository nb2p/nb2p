import re
import tokenize
from io import StringIO
from typing import Dict, Tuple

from tree_sitter import Language, Node, Parser
import tree_sitter_python as tspython

from nb2p import config


def index_to_code_token(index, code_lines) -> str:
    LINE_SEP = "\n"

    start_point = index[0]
    end_point = index[1]
    if start_point[0] == end_point[0]:
        s = code_lines[start_point[0]][start_point[1] : end_point[1]]
    else:
        s = ""
        s += code_lines[start_point[0]][start_point[1] :]
        for i in range(start_point[0] + 1, end_point[0]):
            s += LINE_SEP + code_lines[i]
        s += LINE_SEP + code_lines[end_point[0]][: end_point[1]]
    return s


def node_to_code_token(node: Node, code_lines):
    return index_to_code_token((node.start_point, node.end_point), code_lines)


def pretty_print_sexp(code_lines, node, depth=0):
    if node.is_named:
        print(
            f"{'  ' * depth}({node.type} '{index_to_code_token((node.start_point, node.end_point), code_lines) if node.start_point[0] == node.end_point[0] else '<MULTI_LINE>'}'"
        )
        for child in node.children:
            pretty_print_sexp(code_lines, child, depth + 1)
        print(f"{'  ' * depth})")
    else:
        print(f"{'  ' * depth}{node.type}")


def remove_comments_and_docstrings(source, lang):
    if lang in ["python"]:
        """
        Returns 'source' minus comments and docstrings.
        """
        io_obj = StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            ltext = tok[4]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += " " * (start_col - last_col)
            # Remove comments:
            if token_type == tokenize.COMMENT:
                pass
            # This series of conditionals removes docstrings:
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
                    # This is likely a docstring; double-check we're not inside an operator:
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            out += token_string
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        temp = []
        for x in out.split("\n"):
            if x.strip() != "":
                temp.append(x)
        return "\n".join(temp)
    elif lang in ["ruby"]:
        return source
    else:

        def replacer(match):
            s = match.group(0)
            if s.startswith("/"):
                return " "  # note: a space and not an empty string
            else:
                return s

        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE,
        )
        temp = []
        for x in re.sub(pattern, replacer, source).split("\n"):
            if x.strip() != "":
                temp.append(x)
        return "\n".join(temp)


def tree_to_token_index(root_node):
    if (
        len(root_node.children) == 0
        or root_node.type == "string"
        or root_node.type == "comment"
        or "comment" in root_node.type
    ):
        return [(root_node.start_point, root_node.end_point)]
    else:
        code_tokens = []
        for child in root_node.children:
            code_tokens += tree_to_token_index(child)
        return code_tokens


def tree_to_variable_index(root_node, index_to_code):
    if (
        len(root_node.children) == 0
        or root_node.type == "string"
        or root_node.type == "comment"
        or "comment" in root_node.type
    ):
        index = (root_node.start_point, root_node.end_point)
        _, code = index_to_code[index]
        if root_node.type != code:
            return [(root_node.start_point, root_node.end_point)]
        else:
            return []
    else:
        code_tokens = []
        for child in root_node.children:
            code_tokens += tree_to_variable_index(child, index_to_code)
        return code_tokens


def parser() -> Tuple[Parser, Language]:
    # LANGUAGE = Language(config.parse_path(config.value("parser.path")), "python")
    LANGUAGE = Language(tspython.language())
    parser = Parser(LANGUAGE)
    # parser.set_language(LANGUAGE)

    return parser, LANGUAGE


def travel(root_node, index_to_code, tokenizer):
    """Given a AST node, return AST travel sequence using Algo in the paper:
    https://arxiv.org/pdf/2203.03850.pdf
    """
    if (
        len(root_node.children) == 0
        or root_node.type == "string"
        or root_node.type == "comment"
        or "comment" in root_node.type
    ):
        index = (root_node.start_point, root_node.end_point)
        code = index_to_code[index][1]
        return tokenizer.tokenize(code)
    else:
        code_tokens = []
        for child in root_node.children:
            code_tokens += travel(child, index_to_code, tokenizer)
        # remove nodes that have only one children for reducing length
        if len(root_node.children) != 1:
            return (
                ["AST#" + root_node.type.replace("#", "") + "#Left"]
                + code_tokens
                + ["AST#" + root_node.type.replace("#", "") + "#Right"]
            )
        else:
            return code_tokens


def AST(code, lang, tokenizer):
    """Given a code, return its AST flatten sequence"""
    if lang == "php":
        code = "<?php " + code + "?>"
    # remove comment
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:
        pass
    # # parse source code
    # if lang == "csharp":
    #     tree = parsers["c_sharp"].parse(bytes(code, "utf8"))
    # else:
    #     tree = parsers[lang].parse(bytes(code, "utf8"))
    tree = parser().parse(bytes(code, "utf8"))

    # obtain AST sequence
    root_node = tree.root_node
    tokens_index = tree_to_token_index(root_node)
    code = code.split("\n")
    code_tokens = [index_to_code_token(x, code) for x in tokens_index]
    index_to_code = {}
    for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
        index_to_code[index] = (idx, code)

    code_tokens = travel(root_node, index_to_code, tokenizer)
    return code_tokens


def remove_comments(code, lang):
    """return comments and docstrings from a given code"""
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:
        pass
    return code


if __name__ == "__main__":
    """Test case."""

    MAX_LENGTH = 1024

    code = """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
"""

    from transformers import RobertaConfig, RobertaTokenizer

    tokenizer = RobertaTokenizer.from_pretrained("microsoft/unixcoder-base")
    config = RobertaConfig.from_pretrained("microsoft/unixcoder-base")

    ast = AST(code, "python", tokenizer)
    print(f"len(ast): {len(ast)}")

    tokens = (
        [tokenizer.cls_token]
        + ["<encoder-only>"]
        + [tokenizer.sep_token]
        + ast
        + [tokenizer.eos_token]
    )
    tokens = tokens[: MAX_LENGTH - 4]
    tokens_id = tokenizer.convert_tokens_to_ids(ast)

    import torch

    from nb2p.reprmodel.ux import UniXcoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ux_model = UniXcoder("microsoft/unixcoder-base")
    ux_model.to(device)
    ux_model.eval()

    tokens_id = tokens_id + [config.pad_token_id] * (MAX_LENGTH - len(tokens_id))
    X = torch.tensor([tokens_id]).to(device)
    print(X.shape)

    _, embedding = ux_model(X)

    print(embedding.shape)
    print(embedding)
