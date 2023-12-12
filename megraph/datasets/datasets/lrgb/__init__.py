from .. import graph_dataset_manager as manager

from .peptides_functional import PeptidesFunctionalDataset
from .peptides_structural import PeptidesStructuralDataset


for name in ["struct", "func"]:
    dataset_name = f"peptides_{name}"
    meta_data = dict(
        task="gpred",
        inductive=True,
        embed_method=dict(node="mol"),
    )
    if name == "struct":
        meta_data.update(
            dict(
                fn=PeptidesStructuralDataset,
                eval_metric="MAE",
                loss_func="L1",
                reg=True,
            )
        )
    elif name == "func":
        meta_data.update(
            dict(
                fn=PeptidesFunctionalDataset,
                eval_metric="ap",
                loss_func="BCE",
            )
        )
    manager.add_dataset(dataset_name, meta_data)
