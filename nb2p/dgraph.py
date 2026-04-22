"""Dependency graph based on variable references."""

import tempfile
from typing import List, Literal, Optional, Tuple

import networkx as nx
from matplotlib import pyplot as plt
from tree_sitter import Node, Tree

from .astparse import node_to_code_token
from .dataset import render_python_code
from .notebook import Notebook


def def_info_obj(
    node: Node,
    def_type: Literal["variable", "function", "class"],
    code_lines: List[str],
):
    return {
        "type": "def",
        "def_type": def_type,
        "node": node,
        "content": node_to_code_token(node, code_lines),
        "line": node.start_point[0],
        "period": (node.start_point, node.end_point),
    }


def ref_info_obj(source_node: Node, target: dict, code_lines: List[str]):
    target_node = target["node"]
    content = target["content"]
    if target["def_type"] != "variable":
        content += "()"

    return {
        "type": "ref",
        "source": source_node,
        "target": target,
        "content": content,
        "line_source": source_node.start_point[0],
        "line_target": target_node.start_point[0],
        "period": (source_node.start_point, source_node.end_point),
    }


def parse_def(node: Node, root_node: Node, code_lines: List[str], all_defs: dict):
    if node.type == "function_definition":
        func_id_node = node.child_by_field_name("name")
        if func_id_node:
            yield def_info_obj(func_id_node, "function", code_lines)

    if node.type == "class_definition":
        func_id_node = node.child_by_field_name("name")
        if func_id_node:
            yield def_info_obj(func_id_node, "class", code_lines)

    if node.type == "assignment":
        expr_node = node.child_by_field_name("right")
        if not expr_node:
            """PARSE ERROR. BREAK"""
            raise ValueError("Parse error")

        parse_ref(expr_node, code_lines, all_defs)

        curr_block = node.parent.parent  # type:ignore
        if curr_block.id != root_node.id:  # type:ignore
            """Not top-level assignment"""
            return

        id_node = node.child_by_field_name("left")
        if not id_node:
            """PARSE ERROR. BREAK"""
            raise ValueError("Parse error")

        if id_node.type == "identifier":
            yield def_info_obj(id_node, "variable", code_lines)

        if id_node.type == "pattern_list":
            """Destructuring assignment, e.g., `a, b = func()`"""
            # print(id_node.children)
            yield from (
                def_info_obj(child, "variable", code_lines)
                for child in id_node.children
                if child.type == "identifier"
            )


def parse_ref(node: Node, code_lines: List[str], all_defs: dict):
    var_name = None
    var_node = None

    if node.type == "identifier":
        var_name = node_to_code_token(node, code_lines)
        var_node = node

    if node.type == "object":
        subscript = node.child_by_field_name("subscript")
        if subscript is not None:
            var_node = subscript.child_by_field_name("value")
            if var_node is not None:
                var_name = node_to_code_token(var_node, code_lines)

    if var_name:
        if (
            var_name == "inplace"
            and var_node.next_sibling.next_sibling.type == "true"  # type:ignore
        ):
            """Rule: `inplace=True` happened as a keyword argument is highly possible to be a DataFrame transformation.

            Thus, the `function` sibling's `object` node (as variable name) is re-defined.
            """
            var_node = (
                var_node.parent.parent.prev_sibling.child_by_field_name(  # type:ignore
                    "object"
                )
            )
            info_obj = def_info_obj(var_node, "variable", code_lines)  # type:ignore
            yield info_obj
            all_defs[info_obj["content"]] = info_obj
            # print(
            #     f"redefined variable: {info_obj['content']} at line {info_obj['line']}"
            # )
            return

        """The MAIN variable check logic"""
        if var_name in all_defs and all_defs[var_name]["node"].id != node.id:
            yield ref_info_obj(node, all_defs[var_name], code_lines)

    if var_node:
        if var_node.parent.type == "argument_list" or (  # type:ignore
            var_node.parent.parent is not None  # type:ignore
            and var_node.parent.parent.type == "argument_list"  # type:ignore
        ):
            """Rule:
            If
                1. referenced as an argument, and
                2. function is defined by user (i.e., not package API call)
            Then
                assume side effect exists within function call

            Thus, the reference variable is re-defined.
            """
            if var_node.parent.type == "argument_list":  # type:ignore
                args_node = var_node.parent
            else:
                args_node = var_node.parent.parent  # type:ignore

            func_node = args_node.parent.child_by_field_name(  # type:ignore
                "function"
            )

            if (
                func_node
                and node_to_code_token(func_node, code_lines) in all_defs
                and var_name in all_defs
                and all_defs[var_name]["def_type"] == "variable"
            ):
                # if var_name in all_defs and all_defs[var_name]["def_type"] == "variable":
                info_obj = def_info_obj(var_node, "variable", code_lines)
                yield info_obj
                all_defs[var_name] = info_obj


def traverse_get_vardep(tree: Tree, code_lines: List[str]):
    """Pre-order Traversal of tree nodes."""
    cursor = tree.walk()

    all_defs = {}

    reached_root = False
    while reached_root == False:
        """TT: Yield current node"""
        # yield cursor.node
        curr_node = cursor.node
        defs = list(parse_def(curr_node, tree.root_node, code_lines, all_defs))
        if defs:
            for d in defs:
                all_defs[d["content"]] = d
        yield from defs

        if curr_node.type != "assignment":
            refs = parse_ref(curr_node, code_lines, all_defs)
            yield from refs

        """TT: Yield first child (Depth += 1)"""
        if cursor.goto_first_child():
            continue

        """TT: Yield other children (Same depth)"""
        if cursor.goto_next_sibling():
            continue

        """TT: Returning to parent. If the last child, return again"""
        retracing = True
        while retracing:
            if not cursor.goto_parent():
                """TT: Depth = 1, i.e., the moment from root_node_children[-1] to root"""
                retracing = False
                reached_root = True

            if cursor.goto_next_sibling():
                """TT: Have other children at parent depth. Go in"""
                retracing = False


class DGraph:
    def __init__(self, notebook: Notebook, parser) -> None:
        self._notebook = notebook
        self._parser = parser
        pass

    @property
    def segment_dep(self):
        var_refs = {}

        for vd in self.vardeps:
            if vd["type"] == "ref":
                source_id = self._notebook.segment_id(vd["line_source"])
                target_id = self._notebook.segment_id(vd["line_target"])
                if source_id != target_id:
                    if (source_id, target_id) not in var_refs:
                        var_refs[(source_id, target_id)] = set()

                    var_refs[(source_id, target_id)].add(vd["content"])

        return var_refs

    @property
    def cell_dep(self):
        var_refs = {}

        for vd in self.vardeps:
            if vd["type"] == "ref":
                source_id = self._notebook.code_cell_id(vd["line_source"])
                target_id = self._notebook.code_cell_id(vd["line_target"])
                if source_id != target_id:
                    if (source_id, target_id) not in var_refs:
                        var_refs[(source_id, target_id)] = set()

                    var_refs[(source_id, target_id)].add(vd["content"])

        return var_refs

    @property
    def vardeps(self):
        parse_tree = self._parser.parse("\n".join(self._notebook.code_lines).encode())
        return list(traverse_get_vardep(parse_tree, self._notebook.code_lines))

    @property
    def density(self) -> float:
        return nx.density(self.get_nx_multigraph())

    def get_nx(
        self, mode: Literal["cell", "segment"] = "segment", ref_to_def: bool = False
    ) -> nx.DiGraph:
        g = nx.DiGraph()

        if mode == "segment":
            var_refs = self.segment_dep
            g.add_nodes_from(range(len(self._notebook.segments)))
        else:
            var_refs = self.cell_dep
            g.add_nodes_from(range(len(self._notebook.code_cells)))

        for (src, tgt), vars in var_refs.items():
            g.add_edge(src, tgt, label=vars)

        if not ref_to_def:
            g = g.reverse()

        return g

    def get_nx_multigraph(self, ref_to_def: bool = False) -> nx.MultiDiGraph:
        var_refs = self.segment_dep

        g = nx.MultiDiGraph()
        g.add_nodes_from(range(len(self._notebook.segments)))
        for (src, tgt), vars in var_refs.items():
            for var in vars:
                g.add_edge(src, tgt, label=var)

        if not ref_to_def:
            g = g.reverse()

        return g

    def display_nx(
        self,
        mode: Literal["cell", "segment"] = "segment",
        figsize: Tuple[int, int] = (12, 5),
    ):
        plt.figure(1, figsize=figsize)

        # Generate a left-to-right layout for the nodes
        if mode == "segment":
            nodes = self._notebook.segments
        else:
            nodes = self._notebook.code_cells
        pos_dict = {i: [i, 0] for i in range(0, len(nodes))}

        # Draw the pipeline with curved edges
        nx.draw_networkx(
            self.get_nx(mode=mode),
            pos_dict,
            with_labels=True,
            node_size=500,
            font_size=12,
            node_color="lightblue",
            edge_color="gray",
            arrowsize=20,
            connectionstyle="arc3, rad=0.5",
        )

        plt.xlim(-1, len(nodes))
        plt.ylim(-1, 1)  # Centralize Y-Axis
        plt.show()

    def get_pydot(self):
        return self._do_get_pydot(self.get_nx())

    def get_pydot_multigraph(self):
        return self._do_get_pydot(self.get_nx_multigraph())

    def _do_get_pydot(self, nx_graph):
        graph = nx.drawing.nx_pydot.to_pydot(nx_graph)
        graph.set("splines", "spline")
        graph.set("rankdir", "TB")
        return graph

    def display_pydot(self, with_code: bool = False):
        from IPython.display import Image, display  # type:ignore

        img = self.get_pydot_image(with_code=with_code)
        display(Image(img))

    def get_pydot_image(self, with_code: bool = False):
        graph, _ = self._generate_figure(self.get_pydot(), with_code)
        return graph.create_png()

    def display_pydot_multigraph(self, with_code: bool = False):
        from IPython.display import Image, display  # type:ignore

        graph, _ = self._generate_figure(self.get_pydot_multigraph(), with_code)
        plt = Image(graph.create_png())
        display(plt)

    def _generate_figure(self, graph, with_code: bool):
        if not with_code:
            return graph, []
        else:
            temp_files = []

            for node in graph.get_nodes():
                segment_id = int(node.get_name())
                image_data = render_python_code(
                    "\n".join(self._notebook.segments[segment_id].code_lines)
                )

                f = tempfile.NamedTemporaryFile("wb")
                f.write(image_data)
                f.flush()
                temp_files.append(f)

                node.set("image", f.name)
                node.set("shape", "box")
                node.set("labelloc", "b")
                node.set("imagepos", "tc")

            return graph, temp_files

    def save_pydot(self, path: str, with_code: bool = False):
        graph, temp_files = self._generate_figure(self.get_pydot(), with_code)
        graph.write_pdf(path)
        for f in temp_files:
            f.close()

    def get_dagre_data(self):
        """Generate data structure used for Dagre rendering"""
        result = {"nodes": [], "links": []}

        g = self.get_nx()

        result["nodes"] = [{"id": str(n), "label": str(n)} for n in g.nodes]
        # result['links'] = [{"source": str(s), "target": str(t)} for s, t in g.edges]
        result["links"] = [
            {"source": str(s), "target": str(t), "label": str(l)}
            for (s, t), l in nx.get_edge_attributes(g, "label").items()
        ]

        return result


import networkx as nx
from nb2p import astparse, dgraph
from nb2p.notebook import Notebook
def graph_edit_distance(
    G1,
    G2,
    node_match=None,
    edge_match=None,
    node_subst_cost=None,
    node_del_cost=None,
    node_ins_cost=None,
    edge_subst_cost=None,
    edge_del_cost=None,
    edge_ins_cost=None,
    roots=None,
    upper_bound=None,
    timeout=None,
):
    bestcost = None
    count = 0
    for _, _, cost in nx.optimize_edit_paths(
        G1,
        G2,
        node_match,
        edge_match,
        node_subst_cost,
        node_del_cost,
        node_ins_cost,
        edge_subst_cost,
        edge_del_cost,
        edge_ins_cost,
        upper_bound,
        True,
        roots,
        timeout,
    ):
        # assert bestcost is None or cost < bestcost
        bestcost = cost
        count = count + 1
        if (count > 5):
            break
    return bestcost
    # return bestcost, count

def do_compute_ged_v1(item):

    k, v = item
    
    parser, lang = astparse.parser()
    
    nb = Notebook.from_segment_ends_ast(v['gt_ast'], code=v['code'])
    nb_pred = Notebook.from_segment_ends_ast(v['pred_ast'], code=v['code'])
    # print(nb.segment_ends_ast, nb_pred.segment_ends_ast)

    G1 = dgraph.DGraph(nb, parser)
    G2 = dgraph.DGraph(nb_pred, parser)
    try:
        ged = graph_edit_distance(
            G1.get_nx(), 
            G2.get_nx(), 
            timeout=2, 
            node_match=lambda n1, n2: True, 
            edge_match=lambda e1, e2: e1['label'] == e2['label'],
        )
    except Exception as e:
        print(f"WARN  {e}")
        return None

    return ged
        
def do_compute_ged(item):
    import networkx as nx
    from nb2p import astparse, dgraph
    from nb2p.notebook import Notebook
    import random
    random.seed(123)        # or any integer
    import numpy
    numpy.random.seed(123)
    import os
    os.environ['PYTHONHASHSEED'] = '123'
    k, v = item
    
    parser, lang = astparse.parser()
    
    nb = Notebook.from_segment_ends_ast(v['gt_ast'], code=v['code'])
    nb_pred = Notebook.from_segment_ends_ast(v['pred_ast'], code=v['code'])
    # print(nb.segment_ends_ast, nb_pred.segment_ends_ast)
    G1 = dgraph.DGraph(nb, parser)
    G2 = dgraph.DGraph(nb_pred, parser)
    try:
        G1_nx = G1.get_nx()
        G2_nx = G2.get_nx()
        G1_nx = nx.relabel_nodes(G1_nx, dict(zip(sorted(G1_nx.nodes()), range(len(G1_nx.nodes())))))
        G2_nx = nx.relabel_nodes(G2_nx, dict(zip(sorted(G2_nx.nodes()), range(len(G2_nx.nodes())))))
        ged = nx.graph_edit_distance(
            # G1.get_nx(), 
            # G2.get_nx(), 
            G1_nx,
            G2_nx,
            timeout=1, 
            node_match=lambda n1, n2: True, 
            edge_match=lambda e1, e2: e1['label'] == e2['label'],
        )
    except Exception as e:
        print(f"WARN  {e}")
        return None

    return ged