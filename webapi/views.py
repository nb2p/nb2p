import base64
import logging
import os
from typing import List, Optional

from django.http import HttpResponse, JsonResponse
from tree_sitter import Language, Parser

from nb2p import config, dgraph
from nb2p.notebook import Notebook
from webcommon import code_repr, seg_trans

global cr
global sts
global parser


logger = logging.getLogger(__name__)

conf = config.default_config

sts: dict[str, Optional[seg_trans.SegmentTransitionModel]]


def load_model():
    global cr
    global sts
    global parser
    cr = code_repr.CRModel(conf["code_repr"])
    cr.load_model()

    st_configs = conf["seg_trans"]
    sts = {
        "Decision Tree": seg_trans.DecisionTreeSTModel(st_configs["dtree"]),
        "Random Forest": seg_trans.RandomForestSTModel(st_configs["rforest"]),
        "XGBoost": seg_trans.XGBoostSTModel(st_configs["xgboost"]),
        "Transformer": seg_trans.TransformerSTModel(st_configs["transformer"]),
        "NB2P-SS": seg_trans.NB2PSSSTModel(st_configs["nb2pss"]),
    }

    for k, v in sts.items():
        if v and v.enabled:
            v.load_model()
        else:
            """If not enabling the model, remove the object to save memory"""
            sts[k] = None

    parser = Parser()
    parser.set_language(Language("./build/tree-sitter-python.so", "python"))


load_model()


def transform_code(code_lines: List[str]):
    return [l for l in code_lines if not l.startswith("#") and not l.strip() == ""]


def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")


def ping(request):
    return JsonResponse({"msg": "pong"})


def resolve_notebook_path(notebook_path: str):
    if not notebook_path.startswith("/"):
        notebook_path = os.path.abspath(
            os.path.join(os.getcwd(), "web/public", notebook_path)
        )
    return notebook_path


def notebook(request):
    notebook_path: str = request.POST.get("path")
    logger.debug(f"Request notebook: {notebook_path}")

    notebook_path = resolve_notebook_path(notebook_path)
    logger.debug(f"Resolved notebook path: {notebook_path}")

    with open(notebook_path) as f:
        data = f.read()

    return JsonResponse({"path": notebook_path, "notebookStr": data})


def notebook_code(request):
    notebook_path = request.POST.get("path")
    logger.debug(f"Request code for notebook: {notebook_path}")
    if notebook_path is None:
        return JsonResponse({"err": "Invalid notebook path"}, status=400)

    notebook_path = resolve_notebook_path(notebook_path)
    logger.debug(f"Resolved notebook path: {notebook_path}")

    try:
        notebook_df = code_repr.NotebookLoader(notebook_path).load()
        if notebook_df is None:
            return JsonResponse({"err": "Cannot load notebook"}, status=400)

        notebook = Notebook.from_dataframe(notebook_df).remove_code_cell_comments()
    except FileNotFoundError as e:
        return JsonResponse({"err": e.strerror}, status=400)

    code = "\n".join(transform_code(notebook.code_lines))
    # code = "\n".join(code_lines)

    return JsonResponse({"path": notebook_path, "code": code})


def notebook_cells(request):
    notebook_path = request.POST.get("path")
    logger.debug(f"Request cells for notebook: {notebook_path}")
    if notebook_path is None:
        return JsonResponse({"err": "Invalid notebook path"}, status=400)

    notebook_path = resolve_notebook_path(notebook_path)
    logger.debug(f"Resolved notebook path: {notebook_path}")

    try:
        notebook_df = code_repr.NotebookLoader(notebook_path).load()
        if notebook_df is None:
            return JsonResponse({"err": "Cannot load notebook"}, status=400)

        notebook = Notebook.from_dataframe(notebook_df).remove_code_cell_comments()
    except FileNotFoundError as e:
        return JsonResponse({"err": e.strerror}, status=400)

    logger.debug(f"Cells for notebook {notebook_path}: {notebook.code_cells}")

    # cells = [transform_code(c) for c in cells]
    # # code = "\n".join(transform_code(code_lines))
    # # code = "\n".join(code_lines)

    return JsonResponse(
        {"path": notebook_path, "cells": [cell.content for cell in notebook.code_cells]}
    )


def notebook_segment_ends(request):
    code_lines = []

    notebook_path = request.POST.get("path")
    if notebook_path is not None:
        logger.debug(f"Request segment ends for notebook: {notebook_path}")

        notebook_path = resolve_notebook_path(notebook_path)
        logger.debug(f"Resolved notebook path: {notebook_path}")

        try:
            notebook_df = code_repr.NotebookLoader(notebook_path).load()
            if notebook_df is None:
                return JsonResponse({"err": "Cannot load notebook"}, status=400)

            notebook = Notebook.from_dataframe(notebook_df).remove_code_cell_comments()
            code_lines = notebook.code_lines
        except FileNotFoundError as e:
            return JsonResponse({"err": e.strerror}, status=400)
    else:
        notebook_code = request.POST.get("code")
        if notebook_code is not None:
            logger.debug(f"Request segment ends for notebook code: {notebook_path}")
            code_lines = notebook_code.split("\n")
            notebook = Notebook.from_code_cells([code_lines])
        else:
            return JsonResponse(
                {"err": "Need either notebook path or code"}, status=400
            )

    embedding, ast_cl_map = cr.get_embedding(code_lines=code_lines)
    # embedding = cr.get_embedding(code_lines=code_lines)
    if embedding is None or ast_cl_map is None:
        logging.warning(f"Notebook cannot be parsed: {notebook_path}")
        return JsonResponse(
            {"err": "Notebook cannot be parsed", "path": notebook_path}, status=400
        )

    stmodel_id = request.POST.get("stmodel")
    if not (stmodel_id and stmodel_id in sts and sts[stmodel_id]):
        return JsonResponse(
            {"err": "Segment transition model not available", "stmodel": stmodel_id},
            status=404,
        )
    stmodel = sts[stmodel_id]
    assert stmodel is not None

    ast_ends, segment_ends = stmodel.segment_ends(embedding, ast_cl_map)
    logging.debug(f"AST ends for notebook using ST model {stmodel_id}: {ast_ends}")
    logging.debug(
        f"Segment ends for notebook using ST model {stmodel_id}: {segment_ends}"
    )

    """Draw Graph"""
    nb = Notebook.from_segment_ends(segment_ends, notebook, mode="line")
    # from pprint import pprint

    # pprint([s.code_lines for s in nb.segments])
    G = dgraph.DGraph(nb, parser)
    image_data = base64.b64encode(G.get_pydot_image()).decode("utf-8")
    code_image_data = base64.b64encode(G.get_pydot_image(with_code=True)).decode(
        "utf-8"
    )

    return JsonResponse(
        {
            "path": notebook_path,
            "segmentEnds": segment_ends,
            "pipelineBase64": image_data,
            "pipelineCodeBase64": code_image_data,
        }
    )


def notebook_segment_ends_v2(request):
    code_lines = []

    notebook_path = request.POST.get("path")
    if notebook_path is not None:
        logger.debug(f"Request segment ends for notebook: {notebook_path}")

        notebook_path = resolve_notebook_path(notebook_path)
        logger.debug(f"Resolved notebook path: {notebook_path}")

        try:
            notebook_df = code_repr.NotebookLoader(notebook_path).load()
            if notebook_df is None:
                return JsonResponse({"err": "Cannot load notebook"}, status=400)

            notebook = Notebook.from_dataframe(notebook_df).remove_code_cell_comments()
            code_lines = notebook.code_lines
        except FileNotFoundError as e:
            return JsonResponse({"err": e.strerror}, status=400)
    else:
        notebook_code = request.POST.get("code")
        if notebook_code is not None:
            logger.debug(f"Request segment ends for notebook code: {notebook_path}")
            code_lines = notebook_code.split("\n")
            notebook = Notebook.from_code_cells([code_lines])
        else:
            return JsonResponse(
                {"err": "Need either notebook path or code"}, status=400
            )

    embedding, ast_cl_map = cr.get_embedding(code_lines=code_lines)
    # embedding = cr.get_embedding(code_lines=code_lines)
    if embedding is None or ast_cl_map is None:
        logging.warning(f"Notebook cannot be parsed: {notebook_path}")
        return JsonResponse(
            {"err": "Notebook cannot be parsed", "path": notebook_path}, status=400
        )

    stmodel_id = request.POST.get("stmodel")
    if not (stmodel_id and stmodel_id in sts and sts[stmodel_id]):
        return JsonResponse(
            {"err": "Segment transition model not available", "stmodel": stmodel_id},
            status=404,
        )
    stmodel = sts[stmodel_id]
    assert stmodel is not None

    ast_ends, segment_ends = stmodel.segment_ends(embedding, ast_cl_map)
    logging.debug(f"AST ends for notebook using ST model {stmodel_id}: {ast_ends}")
    logging.debug(
        f"Segment ends for notebook using ST model {stmodel_id}: {segment_ends}"
    )

    """Draw Graph"""
    nb = Notebook.from_segment_ends(segment_ends, notebook, mode="line")

    G = dgraph.DGraph(nb, parser)
    pipeline_display_data = G.get_dagre_data()

    return JsonResponse(
        {
            "path": notebook_path,
            "segmentEnds": segment_ends,
            "pipeline": pipeline_display_data,
        }
    )
