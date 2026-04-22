from typing import List
from nb2p import npop
import numpy as np
from dfgtree import DFGNode
import queue
import itertools
from itertools import chain
from collections.abc import Iterable
import torch
from torch import nn


def clean_segment_ends(segment_ends: List[int], max_length: int):
    result = set()
    for x in segment_ends:
        if x >= max_length:
            return None, f"segment ends exceed max length: {x}"
        if x not in result and x >= 0:
            result.add(x)

    result = sorted(result)

    if len(result) == 0:
        return None, "empty segment ends after cleaning"

    return result, None


def build_shallow_input(sample_dict: dict, max_length: int):
    sample_dict["segment_ends"], err = clean_segment_ends(
        sample_dict["segment_ends"], max_length
    )
    if err:
        # print(f"WARN  ignore {sample}. Reason: {err}")
        raise ValueError(err)

    sample_dict["y"] = npop.indices_to_binary(
        sample_dict["segment_ends"], sample_dict["segment_ends"][-1] + 1
    )

    func_encodings = []
    for fdef in sample_dict["func_defs"]:
        func_encodings.append(fdef["repr"])

    encodings = []
    for s in sample_dict["segments"]:
        encodings.extend([c.repr[0] for c in s["repr"].children])

    if len(encodings) == 0:
        # print(f"WARN  ignore {sample}. Reason: encodings is empty")
        raise ValueError("encoding is empty")

    sample_dict["x"] = np.array(encodings)

    code = "\n".join([s['code'] for s in sample_dict['segments']])

    del sample_dict["segments"]
    del sample_dict["func_defs"]
    
    sample_dict["gt_ast"] = sample_dict["segment_ends"]
    sample_dict['code'] = code

    return sample_dict

################################################################################

def replace_elements(data, replacement_function):
    """
    Recursively replaces elements in an arbitrarily nested list based on a replacement function.

    Args:
        data: The input data, which can be a list or any other iterable.
        replacement_function: A function that takes an element and returns its replacement.

    Returns:
        A new list with replaced elements.
    """

    if not isinstance(data, list):
        return replacement_function(data)

    result = []
    for element in data:
        result.append(replace_elements(element, replacement_function))
    return result


def flatten(xs):
    for x in xs:
        # print(x)
        if isinstance(x, tuple):
            for node in x[1]:
                yield (x[0], node)
        elif isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x

            
def bfs(root: DFGNode):
  """Performs Breadth-First Search on a DFGNode tree.

  Args:
    root: The root node of the DFGNode tree.

  Yields:
    A tuple containing the node's representation and its depth from the root.
  """

  q = queue.Queue()
  q.put((root, 0))

  while not q.empty():
    node, depth = q.get()
    yield node.repr, depth

    for child in node.children:
      q.put((child, depth + 1))


def bfs_layered(root: DFGNode):
    """Performs Breadth-First Search on a DFGNode tree, returning only the layer representations.

    Args:
        root: The root node of the DFGNode tree.

    Returns:
        A list of layers, where each layer is a list of node representations at that depth.
    """

    layers = []
    current_layer = [(0, [root])]
    next_layer = []

    while current_layer:
        nodes = list(flatten([current_layer]))
        if len(nodes) == 0:
            break

        layers.append(current_layer)

        for i, (last_layer_i, node) in enumerate(nodes):
            if node.children:
                next_layer.append((i, node.children))

        current_layer = next_layer
        next_layer = []

    return layers


def build_deep_input(sample_dict: dict, max_length: int, device: torch.device):
    sample_dict["segment_ends"], err = clean_segment_ends(
        sample_dict["segment_ends"], max_length
    )
    if err:
        print(f"WARN  ignore {sample}. Reason: {err}")
        return None

    sample_dict["y"] = npop.indices_to_binary(
        sample_dict["segment_ends"], sample_dict["segment_ends"][-1] + 1
    )
    
    # extract encodings
    ast_children = []
    for s in sample_dict['segments']:
        node = s['repr']
        ast_children.extend(node.children)

    bfs_result = list(bfs_layered(DFGNode(None, ast_children)))
    bfs_result = bfs_result[1:]
    # pprint(bfs_result)

    if len(bfs_result) == 0:
        # print(f"WARN  ignore {sample}. Reason: encodings is empty")
        return None

    result = list(replace_elements(bfs_result, lambda x: [(x[0], torch.Tensor(b.repr).to(device)) for b in x[1]]))
    # print(result)
    result = [list(itertools.chain.from_iterable(r)) for r in result]

    code = "\n".join([s['code'] for s in sample_dict['segments']])
    
    return {
        "code": code,
        "gt_ast": sample_dict["segment_ends"],
        "x": result,
        "y": sample_dict["y"]
    }