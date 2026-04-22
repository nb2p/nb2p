"""Token utilities."""

from typing import List, Optional

# import code_tokenize as ctok


def flatten(A):
    rt = []
    for i in A:
        if isinstance(i, list):
            rt.extend(flatten(i))
        else:
            rt.append(i)
    return rt


def ngrams(input_list, n):
    return list(zip(*[input_list[i:] for i in range(n)]))


def get_tokens_roberta(
    code_lines: List[str],
    tokenizer,
    flat_inner_most=False,
):
    lines_tokens = [tokenizer.tokenize(line) for line in code_lines if line]
    if flat_inner_most:
        return [token for tokens in lines_tokens for token in tokens]
    else:
        return lines_tokens


# def get_tokens_ctok(code_lines: List[str], flat_inner_most=False):
#     lines_tokens = [
#         [str(token)]
#         for line in code_lines
#         if line.strip()
#         for token in ctok.tokenize(line, lang="python", syntax_error="ignore")
#     ]
#     if flat_inner_most:
#         return [token for tokens in lines_tokens for token in tokens]
#     else:
#         return lines_tokens


# def notebook_tokens(
#     code_cells: List[List[str]],
#     flat_inner_most=False,
#     n_gram: Optional[int] = None,
# ):
#     if n_gram:
#         return [ngrams(get_tokens_ctok(c, flat_inner_most), n_gram) for c in code_cells]
#     else:
#         return [get_tokens_ctok(c, flat_inner_most) for c in code_cells]
