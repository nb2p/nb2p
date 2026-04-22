import argparse
import glob
import logging
import os

import conf

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser("scs-cli", "Source Code Segmentation CLI")
parser.add_argument("-c", "--config", help="Config file path", required=True)
parser.add_argument("-n", "--notebook", help="Notebook file path", required=True)

args = parser.parse_args()
conf = conf.load(args.config)

notebooks = []

if os.path.isdir(args.notebook):
    logging.info("Notebook path is directory. Retrieving all notebooks in it")
    notebooks = glob.glob(os.path.join(args.notebook, "*.ipynb"))
else:
    notebooks = [args.notebook]

import code_repr
import seg_trans

cr = code_repr.CRModel(conf["code_repr"])
cr.load_model()

st = seg_trans.SegmentTransitionModel(conf["seg_trans"])
st.load_model()

for nb in notebooks:
    code_lines = cr.get_code_lines(nb)
    if not code_lines:
        logging.warning(f"Notebook cannot be parsed: {nb}")
        continue

    code = "\n".join(code_lines)
    # logging.info(f"Code in {nb}: {code}")

    embedding, ast_cl_map = cr.get_embedding(nb)
    if embedding is None or ast_cl_map is None:
        logging.warning(f"Notebook cannot be parsed: {nb}")
        continue

    # print(embedding)
    # print(st.predict_proba(embedding))
    # print(st.predict(embedding))
    logging.info(
        f"Segment ends for notebook {nb}: {st.segment_ends(embedding, ast_cl_map)}"
    )
