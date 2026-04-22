import pymongo
from nb2p.dfgtree.dfg import CodeEncodingBuilder
from nb2p.dfgtree.dfgtree import DFGTree, preprocess
from typing import List
from nb2p.astparse import Parser, Language
from nb2p.notebook import Notebook

from tqdm import tqdm
import ctypes
import gc
import torch
import chromadb

def to_chromadb(
    db: pymongo.database.Database,
    collection: chromadb.Collection,
    builder: CodeEncodingBuilder,
    parser: Parser,
    lang: Language,
    ids: List[int],
):
    INSERT_INTERVAL = 1000

    ignored = []
    pbar = tqdm(enumerate(ids), total=len(ids))

    documents=[]
    metadatas=[]
    insert_ids=[]

    def insert_into_db():
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=insert_ids,
        )

        documents.clear()
        metadatas.clear()
        insert_ids.clear()

    
    for i, nb_id in pbar:
        nb_data = db.notebooksegments.find_one({"_id": nb_id})
        if nb_data is None:
            ignored.append(nb_id)
            print(f"WARN  no such notebook: {nb_id}")
            continue

        nb_id = str(nb_id)

        if len(collection.get(ids=[nb_id], include=["metadatas"])['ids']) > 0:
            ignored.append(nb_id)
            continue
        
        nb = Notebook.from_db_result(nb_data)
        try:
            pr = preprocess(nb, parser, lang)
        except Exception as e:
            ignored.append(nb_id)
            print(nb_id, e)
            continue
        
        code = "\n".join(pr['segments'])
        segment_ends = nb_data['encoding']['segment_ends']
        
        documents.append(code)
        metadatas.append({"segment_ends": str(segment_ends)})
        insert_ids.append(nb_id)

        if len(documents) >= INSERT_INTERVAL:
            insert_into_db()

            torch.cuda.empty_cache()
            gc.collect()
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)

        pbar.set_postfix(
            {
                "n_ignored": len(ignored),
                # "n_seg": len(segment_ends),
            }
        )

    if len(documents) > 0:
        insert_into_db()

    return ignored