# From https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/data/SBMs.py

import os
import pickle

import dgl
import numpy as np
from dgl import backend as F
from dgl.data.dgl_dataset import DGLDataset
from dgl.data.utils import download, load_graphs, save_graphs
from tqdm import tqdm


class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


class SBMsDatasetDGL(DGLDataset):
    def __init__(
        self,
        name,
        raw_dir=None,
        force_reload=False,
        verbose=False,
        transform=None,
    ):
        assert name in ["SBM_PATTERN", "SBM_CLUSTER"]
        if name == "SBM_PATTERN":
            self._url = "https://www.dropbox.com/s/qvu0r11tjyt6jyb/SBM_PATTERN.zip?dl=1"
        elif name == "SBM_CLUSTER":
            self._url = "https://www.dropbox.com/s/e67bisl7zpqnioq/SBM_CLUSTER.zip?dl=1"
        super().__init__(
            name=name,
            url=self._url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )

    def has_cache(self):
        graph_path = os.path.join(self.save_path, "dgl_graph.bin")
        return os.path.exists(graph_path)

    def download(self):
        file_path = os.path.join(self.raw_dir, f"{self._name}.zip")
        if not os.path.exists(file_path):
            download_cmd = f"curl {self._url} -o {file_path} -J -L -k"
            os.system(download_cmd)
        if not os.path.exists(os.path.join(self.raw_dir, f"{self._name}_train.pkl")):
            self.unzip_file(file_path)

    def unzip_file(self, file_path):
        import zipfile

        zFile = zipfile.ZipFile(file_path, "r")
        for fileM in zFile.namelist():
            if "__MACOSX" in fileM:
                continue
            zFile.extract(fileM, self.raw_dir)
        zFile.close()

    def prepare(self):
        test_file = open(os.path.join(self.raw_dir, f"{self._name}_test.pkl"), "rb")
        self.test_dataset = pickle.load(test_file)

        val_file = open(os.path.join(self.raw_dir, f"{self._name}_val.pkl"), "rb")
        self.val_dataset = pickle.load(val_file)

        train_file = open(os.path.join(self.raw_dir, f"{self._name}_train.pkl"), "rb")
        self.train_dataset = pickle.load(train_file)

        self.dataset = self.test_dataset + self.val_dataset + self.train_dataset
        self.n_samples = len(self.dataset)

    def process(self):
        self.prepare()
        print(f"processing {self.n_samples} graphs")
        self.graphs = []
        for data in tqdm(self.dataset):
            node_features = data.node_feat
            edge_list = (data.W != 0).nonzero()  # converting adj matrix to edge_list

            # Create the DGL Graph
            g = dgl.DGLGraph()
            g.add_nodes(node_features.size(0))
            g.ndata["feat"] = node_features.long()
            for src, dst in edge_list:
                g.add_edges(src.item(), dst.item())
            g.ndata["label"] = F.tensor(
                data.node_label, dtype=F.data_type_dict["int64"]
            )

            # adding edge features for Residual Gated ConvNet
            # edge_feat_dim = g.ndata['feat'].size(1) # dim same as node feature dim
            edge_feat_dim = 1  # dim same as node feature dim
            g.edata["feat"] = F.tensor(np.ones((g.number_of_edges(), edge_feat_dim)))

            self.graphs.append(g)

    def __getitem__(self, idx):
        r"""Get graph and label by index

        Parameters
        ----------
        idx : int
            Item index

        Returns
        -------
        (:class:`dgl.DGLGraph`, Tensor)
        """
        if self._transform is None:
            return self.graphs[idx]
        else:
            return self._transform(self.graphs[idx])

    def save(self):
        """save the graph list and the labels"""
        graph_path = os.path.join(self.save_path, "dgl_graph.bin")
        save_graphs(str(graph_path), self.graphs)

    def load(self):
        graphs = load_graphs(os.path.join(self.save_path, "dgl_graph.bin"))[0]
        self.graphs = graphs

    def get_idx_split(self):
        if self._name == "SBM_PATTERN":
            test_split, val_split, train_split = 2000, 2000, 10000
        elif self._name == "SBM_CLUSTER":
            test_split, val_split, train_split = 1000, 1000, 10000

        test_idx = np.arange(test_split)
        valid_idx = np.arange(val_split) + test_split
        train_idx = np.arange(train_split) + (test_split + val_split)

        return {
            "train": F.tensor(train_idx, dtype=F.data_type_dict["int64"]),
            "valid": F.tensor(valid_idx, dtype=F.data_type_dict["int64"]),
            "test": F.tensor(test_idx, dtype=F.data_type_dict["int64"]),
        }

    def statistics(self):
        return None, self.num_classes, None

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graphs)

    @property
    def num_classes(self):
        """Number of labels for each graph, i.e. number of prediction tasks."""
        return F.unique(self.graphs[0].ndata["label"]).size(0)  # 2

    @property
    def node_feat_size(self):
        return F.unique(self.graphs[0].ndata["feat"]).size(0)  # 3


def get_sbm_pattern(raw_dir=None, **kwargs):
    return SBMsDatasetDGL(name="SBM_PATTERN", raw_dir=raw_dir, **kwargs)


def get_sbm_cluster(raw_dir=None, **kwargs):
    return SBMsDatasetDGL(name="SBM_CLUSTER", raw_dir=raw_dir, **kwargs)
