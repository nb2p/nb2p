from argparse import ArgumentParser

from tqdm import tqdm
from nb2p import config, database, fileop
from nb2p.dfgtree.dfgtree import get_num_ast_children
from nb2p.astparse import parser


def do_update(dataset_name: str, dry_run: bool):
    db, client = database.connect(dataset_name=dataset_name)

    p, lang = parser()

    for cell_data in tqdm(db.cell.find({"content_no_comment.0": {"$exists": True}})):
        tree = p.parse(bytes("\n".join(cell_data["content_no_comment"]), "utf8"))
        n_ast_children = get_num_ast_children(tree)

        if not dry_run:
            db.cell.update_one(
                {"_id": cell_data["_id"]},
                {"$set": {"n_ast_children_of_cell": n_ast_children}},
            )
        else:
            print(f"[Worker] {cell_data['_id']}: {n_ast_children}")

    client.close()


if __name__ == "__main__":
    arg = ArgumentParser()
    arg.add_argument("-d", "--dataset", help="dataset name", required=True)
    arg.add_argument("-j", "--job", type=int, help="number of jobs", default=8)
    arg.add_argument("-D", "--dry", action="store_true", help="dry run", default=False)
    args = arg.parse_args()

    do_update(args.dataset, args.dry)
