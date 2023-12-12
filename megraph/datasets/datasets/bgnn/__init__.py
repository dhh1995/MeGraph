from .. import graph_dataset_manager as manager

from .SBMs import get_sbm_pattern, get_sbm_cluster
from .superpixels import get_cifar10, get_mnist
from .molecules import get_zinc, get_zinc_full, get_aqsol

SUPERPIXEL_DATASETS = {
    "cifar10": get_cifar10,
    "mnist": get_mnist,
}
SBM_DATASETS = {
    "sbm_pattern": get_sbm_pattern,
    "sbm_cluster": get_sbm_cluster,
}
MOLECULES_DATASETS = {
    "zinc": get_zinc,
    "zinc_full": get_zinc_full,
    "aqsol": get_aqsol,
}


for name, fn in SUPERPIXEL_DATASETS.items():
    manager.add_dataset(name, dict(fn=fn, task="gpred", inductive=True))

for name, fn in SBM_DATASETS.items():
    manager.add_dataset(
        name,
        dict(
            fn=fn,
            task="npred",
            inductive=True,
            loss_func="WCE",
            eval_metric="SBM_acc",
            embed_method=dict(node="embed"),
        ),
    )

for name, fn in MOLECULES_DATASETS.items():
    manager.add_dataset(
        name,
        dict(
            fn=fn,
            task="gpred",
            inductive=True,
            reg=True,
            loss_func="L1",
            eval_metric="MAE",
            embed_method=dict(node="embed"),
        ),
    )
