import json
import os

import dgl.backend as F
import numpy as np
from dgl import DGLError
from dgl.data import DGLDataset


# Available for DGL >= 0.9, copy here for backward compatibility
# https://github.com/dmlc/dgl/blob/master/python/dgl/data/adapter.py
class AsGraphPredDataset(DGLDataset):
    """Repurpose a dataset for standard graph property prediction task.

    The created dataset will include data needed for graph property prediction.
    Currently it only supports homogeneous graphs.

    The class converts a given dataset into a new dataset object such that:

      - It stores ``len(dataset)`` graphs.
      - The i-th graph and its label is accessible from ``dataset[i]``.

    The class will generate a train/val/test split if :attr:`split_ratio` is provided.
    The generated split will be cached to disk for fast re-loading. If the provided split
    ratio differs from the cached one, it will re-process the dataset properly.

    Parameters
    ----------
    dataset : DGLDataset
        The dataset to be converted.
    split_ratio : (float, float, float), optional
        Split ratios for training, validation and test sets. They must sum to one.

    Attributes
    ----------
    num_tasks : int
        Number of tasks to predict.
    num_classes : int
        Number of classes to predict per task, None for regression datasets.
    train_idx : Tensor
        An 1-D integer tensor of training node IDs.
    val_idx : Tensor
        An 1-D integer tensor of validation node IDs.
    test_idx : Tensor
        An 1-D integer tensor of test node IDs.
    node_feat_size : int
        Input node feature size, None if not applicable.
    edge_feat_size : int
        Input edge feature size, None if not applicable.

    Examples
    --------

    >>> from dgl.data import AsGraphPredDataset
    >>> from ogb.graphproppred import DglGraphPropPredDataset
    >>> dataset = DglGraphPropPredDataset(name='ogbg-molhiv')
    >>> new_dataset = AsGraphPredDataset(dataset)
    >>> print(new_dataset)
    Dataset("ogbg-molhiv-as-graphpred", num_graphs=41127, save_path=...)
    >>> print(len(new_dataset))
    41127
    >>> print(new_dataset[0])
    (Graph(num_nodes=19, num_edges=40,
           ndata_schemes={'feat': Scheme(shape=(9,), dtype=torch.int64)}
           edata_schemes={'feat': Scheme(shape=(3,), dtype=torch.int64)}), tensor([0]))
    """

    def __init__(self, dataset, split_ratio=None, **kwargs):
        self.dataset = dataset
        self.split_ratio = split_ratio
        super().__init__(
            dataset.name + "-as-graphpred",
            hash_key=(split_ratio, dataset.name, "graphpred"),
            **kwargs
        )

    def process(self):
        if self.split_ratio is None:
            if hasattr(self.dataset, "get_idx_split"):
                split = self.dataset.get_idx_split()
                self.train_idx = split["train"]
                self.val_idx = split["valid"]
                self.test_idx = split["test"]
            else:
                # Handle FakeNewsDataset
                try:
                    self.train_idx = F.nonzero_1d(self.dataset.train_mask)
                    self.val_idx = F.nonzero_1d(self.dataset.val_mask)
                    self.test_idx = F.nonzero_1d(self.dataset.test_mask)
                except:
                    raise DGLError(
                        "The input dataset does not have default train/val/test\
                        split. Please specify split_ratio to generate the split."
                    )
        else:
            if self.verbose:
                print("Generating train/val/test split...")
            train_ratio, val_ratio, _ = self.split_ratio
            num_graphs = len(self.dataset)
            num_train = int(num_graphs * train_ratio)
            num_val = int(num_graphs * val_ratio)

            idx = np.random.permutation(num_graphs)
            self.train_idx = F.tensor(idx[:num_train])
            self.val_idx = F.tensor(idx[num_train : num_train + num_val])
            self.test_idx = F.tensor(idx[num_train + num_val :])

        if hasattr(self.dataset, "num_classes"):
            # GINDataset, MiniGCDataset, FakeNewsDataset, TUDataset,
            # LegacyTUDataset, BA2MotifDataset
            self.num_classes = self.dataset.num_classes
        else:
            # None for multi-label classification and regression
            self.num_classes = None

        if hasattr(self.dataset, "num_tasks"):
            # OGB datasets
            self.num_tasks = self.dataset.num_tasks
        else:
            self.num_tasks = 1

    def has_cache(self):
        return os.path.isfile(
            os.path.join(self.save_path, "info_{}.json".format(self.hash))
        )

    def load(self):
        with open(
            os.path.join(self.save_path, "info_{}.json".format(self.hash)), "r"
        ) as f:
            info = json.load(f)
            if info["split_ratio"] != self.split_ratio:
                raise ValueError(
                    "Provided split ratio is different from the cached file. "
                    "Re-process the dataset."
                )
            self.split_ratio = info["split_ratio"]
            self.num_tasks = info["num_tasks"]
            self.num_classes = info["num_classes"]

        split = np.load(os.path.join(self.save_path, "split_{}.npz".format(self.hash)))
        self.train_idx = F.zerocopy_from_numpy(split["train_idx"])
        self.val_idx = F.zerocopy_from_numpy(split["val_idx"])
        self.test_idx = F.zerocopy_from_numpy(split["test_idx"])

    def save(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        with open(
            os.path.join(self.save_path, "info_{}.json".format(self.hash)), "w"
        ) as f:
            json.dump(
                {
                    "split_ratio": self.split_ratio,
                    "num_tasks": self.num_tasks,
                    "num_classes": self.num_classes,
                },
                f,
            )
        np.savez(
            os.path.join(self.save_path, "split_{}.npz".format(self.hash)),
            train_idx=F.zerocopy_to_numpy(self.train_idx),
            val_idx=F.zerocopy_to_numpy(self.val_idx),
            test_idx=F.zerocopy_to_numpy(self.test_idx),
        )

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    @property
    def node_feat_size(self):
        if hasattr(self.dataset, "node_feat_size"):
            return self.dataset.node_feat_size
        else:
            g = self[0][0]
            return g.ndata["feat"].shape[-1] if "feat" in g.ndata else None

    @property
    def edge_feat_size(self):
        if hasattr(self.dataset, "edge_feat_size"):
            return self.dataset.edge_feat_size
        else:
            g = self[0][0]
            return g.edata["feat"].shape[-1] if "feat" in g.edata else None


class InductiveNodePredDataset(DGLDataset):
    """Repurpose a dataset for a standard semi-supervised transductive
    node prediction task.

    The class converts a given dataset into a new dataset object that:

      - Contains only one graph, accessible from ``dataset[0]``.
      - The graph stores:

        - Node labels in ``g.ndata['label']``.
        - Train/val/test masks in ``g.ndata['train_mask']``, ``g.ndata['val_mask']``,
          and ``g.ndata['test_mask']`` respectively.
      - In addition, the dataset contains the following attributes:

        - ``num_classes``, the number of classes to predict.
        - ``train_idx``, ``val_idx``, ``test_idx``, train/val/test indexes.

    If the input dataset contains heterogeneous graphs, users need to specify the
    ``target_ntype`` argument to indicate which node type to make predictions for.
    In this case:

      - Node labels are stored in ``g.nodes[target_ntype].data['label']``.
      - Training masks are stored in ``g.nodes[target_ntype].data['train_mask']``.
        So do validation and test masks.

    The class will keep only the first graph in the provided dataset and
    generate train/val/test masks according to the given spplit ratio. The generated
    masks will be cached to disk for fast re-loading. If the provided split ratio
    differs from the cached one, it will re-process the dataset properly.

    Parameters
    ----------
    dataset : DGLDataset
        The dataset to be converted.
    split_ratio : (float, float, float), optional
        Split ratios for training, validation and test sets. Must sum to one.
    target_ntype : str, optional
        The node type to add split mask for.

    Attributes
    ----------
    num_classes : int
        Number of classes to predict.
    train_idx : Tensor
        An 1-D integer tensor of training node IDs.
    val_idx : Tensor
        An 1-D integer tensor of validation node IDs.
    test_idx : Tensor
        An 1-D integer tensor of test node IDs.

    Examples
    --------
    >>> ds = dgl.data.AmazonCoBuyComputerDataset()
    >>> print(ds)
    Dataset("amazon_co_buy_computer", num_graphs=1, save_path=...)
    >>> new_ds = dgl.data.AsNodePredDataset(ds, [0.8, 0.1, 0.1])
    >>> print(new_ds)
    Dataset("amazon_co_buy_computer-as-nodepred", num_graphs=1, save_path=...)
    >>> print('train_mask' in new_ds[0].ndata)
    True
    """

    def __init__(self, dataset, split_ratio=None, target_ntype=None, **kwargs):
        self.dataset = dataset
        self.split_ratio = split_ratio
        self.target_ntype = target_ntype
        super().__init__(
            self.dataset.name + "-as-nodepred",
            hash_key=(split_ratio, target_ntype, dataset.name, "nodepred"),
            **kwargs
        )

    def process(self):
        if self.split_ratio is None:
            if hasattr(self.dataset, "get_idx_split"):
                split = self.dataset.get_idx_split()
                self.train_idx = split["train"]
                self.val_idx = split["valid"]
                self.test_idx = split["test"]
            else:
                # Handle FakeNewsDataset
                try:
                    self.train_idx = F.nonzero_1d(self.dataset.train_mask)
                    self.val_idx = F.nonzero_1d(self.dataset.val_mask)
                    self.test_idx = F.nonzero_1d(self.dataset.test_mask)
                except:
                    raise DGLError(
                        "The input dataset does not have default train/val/test\
                        split. Please specify split_ratio to generate the split."
                    )
        else:
            if self.verbose:
                print("Generating train/val/test split...")
            train_ratio, val_ratio, _ = self.split_ratio
            num_graphs = len(self.dataset)
            num_train = int(num_graphs * train_ratio)
            num_val = int(num_graphs * val_ratio)

            idx = np.random.permutation(num_graphs)
            self.train_idx = F.tensor(idx[:num_train])
            self.val_idx = F.tensor(idx[num_train : num_train + num_val])
            self.test_idx = F.tensor(idx[num_train + num_val :])

        self.num_classes = getattr(self.dataset, "num_classes", None)
        if self.num_classes is None:
            self.num_classes = self.dataset[0].nodes["label"].shape[-1]

    def has_cache(self):
        return os.path.isfile(
            os.path.join(self.save_path, "graph_{}.bin".format(self.hash))
        )

    def load(self):
        with open(
            os.path.join(self.save_path, "info_{}.json".format(self.hash)), "r"
        ) as f:
            info = json.load(f)
            if info["split_ratio"] != self.split_ratio:
                raise ValueError(
                    "Provided split ratio is different from the cached file. "
                    "Re-process the dataset."
                )
            self.split_ratio = info["split_ratio"]
            self.num_tasks = info["num_tasks"]
            self.num_classes = info["num_classes"]

        split = np.load(os.path.join(self.save_path, "split_{}.npz".format(self.hash)))
        self.train_idx = F.zerocopy_from_numpy(split["train_idx"])
        self.val_idx = F.zerocopy_from_numpy(split["val_idx"])
        self.test_idx = F.zerocopy_from_numpy(split["test_idx"])

    def save(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        with open(
            os.path.join(self.save_path, "info_{}.json".format(self.hash)), "w"
        ) as f:
            json.dump(
                {
                    "split_ratio": self.split_ratio,
                    "num_classes": self.num_classes,
                },
                f,
            )
        np.savez(
            os.path.join(self.save_path, "split_{}.npz".format(self.hash)),
            train_idx=F.zerocopy_to_numpy(self.train_idx),
            val_idx=F.zerocopy_to_numpy(self.val_idx),
            test_idx=F.zerocopy_to_numpy(self.test_idx),
        )

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)
