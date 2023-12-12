# From https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/data/molecules.py

import csv
import os
import pickle

import dgl
import numpy as np
from dgl import backend as F
from dgl.data.dgl_dataset import DGLDataset
from dgl.data.utils import download, load_graphs, save_graphs
from tqdm import tqdm


class MoleculeDatasetDGL(DGLDataset):
    def __init__(
        self,
        name="ZINC",
        raw_dir=None,
        force_reload=False,
        verbose=False,
        transform=None,
        part_data=True,
    ):
        assert name in ["ZINC", "ZINC-full", "AqSol"]
        if name == "ZINC":
            self._url = "https://www.dropbox.com/s/feo9qle74kg48gy/molecules.zip?dl=1"
            self.zip_name = "molecules.zip"
            self.data_dir = "molecules"
        elif name == "ZINC-full":
            self._url = (
                "https://www.dropbox.com/s/grhitgnuuixoxwl/molecules_zinc_full.zip?dl=1"
            )
            self.zip_name = "molecules_zinc_full.zip"
            self.data_dir = "molecules/zinc_full"
        elif name == "AqSol":
            self._url = (
                "https://www.dropbox.com/s/lzu9lmukwov12kt/aqsol_graph_raw.zip?dl=1"
            )
            self.zip_name = "aqsol_graph_raw.zip"
            self.data_dir = "aqsol_graph_raw"
        self.part_data = part_data

        super().__init__(
            name=name,
            url=self._url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )

        if self._name == "AqSol":
            self.num_atom_type = 65
            self.num_bond_type = 5
        else:
            self.num_atom_type = 28
            self.num_bond_type = 4

    def has_cache(self):
        graph_path = os.path.join(self.save_path, "dgl_graph.bin")
        return os.path.exists(graph_path)

    def download(self):
        file_path = os.path.join(self.raw_dir, self.zip_name)
        if not os.path.exists(file_path):
            download_cmd = f"curl {self._url} -o {file_path} -J -L -k"
            os.system(download_cmd)
        if not os.path.exists(os.path.join(self.raw_dir, self.data_dir)):
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
        test_file = open(os.path.join(self.raw_dir, self.data_dir, "test.pickle"), "rb")
        self.test_dataset = pickle.load(test_file)
        if self._name == "ZINC" and self.part_data:
            # download index file from https://raw.githubusercontent.com/graphdeeplearning/benchmarking-gnns/master/data/molecules/test.index to ~/.dgl/molecules/
            test_index_file = open(
                os.path.join(self.raw_dir, "molecules", "test.index"), "r"
            )
            data_idx = [list(map(int, idx)) for idx in csv.reader(test_index_file)]
            self.test_dataset = [self.test_dataset[i] for i in data_idx[0]]

        val_file = open(os.path.join(self.raw_dir, self.data_dir, "val.pickle"), "rb")
        self.val_dataset = pickle.load(val_file)
        if self._name == "ZINC" and self.part_data:
            # download index file from https://raw.githubusercontent.com/graphdeeplearning/benchmarking-gnns/master/data/molecules/val.index to ~/.dgl/molecules/
            val_index_file = open(
                os.path.join(self.raw_dir, "molecules", "val.index"), "r"
            )
            data_idx = [list(map(int, idx)) for idx in csv.reader(val_index_file)]
            self.val_dataset = [self.val_dataset[i] for i in data_idx[0]]

        train_file = open(
            os.path.join(self.raw_dir, self.data_dir, "train.pickle"), "rb"
        )
        self.train_dataset = pickle.load(train_file)
        if self._name == "ZINC" and self.part_data:
            # download index file from https://raw.githubusercontent.com/graphdeeplearning/benchmarking-gnns/master/data/molecules/train.index to ~/.dgl/molecules/
            train_index_file = open(
                os.path.join(self.raw_dir, "molecules", "train.index"), "r"
            )
            data_idx = [list(map(int, idx)) for idx in csv.reader(train_index_file)]
            self.train_dataset = [self.train_dataset[i] for i in data_idx[0]]

        self.dataset = self.test_dataset + self.val_dataset + self.train_dataset
        self.n_samples = len(self.dataset)

    def process(self):
        self.prepare()
        print(f"processing {self.n_samples} graphs")
        if self._name in ["AqSol"]:
            return self._process_AqSol()
        elif self.name in ["ZINC", "ZINC-full"]:
            return self._process_zinc()

    def _process_AqSol(self):
        self.graphs = []
        self.label = []
        for dataset in [self.test_dataset, self.val_dataset, self.train_dataset]:
            count_filter1, count_filter2 = 0, 0
            for molecule in tqdm(dataset):
                node_features = F.tensor(molecule[0], dtype=F.data_type_dict["int64"])
                edge_features = F.tensor(molecule[1], dtype=F.data_type_dict["int64"])

                # Create the DGL Graph
                g = dgl.graph((molecule[2][0], molecule[2][1]))

                if g.num_nodes() == 0:
                    count_filter1 += 1
                    continue  # skipping graphs with no bonds/edges

                if g.num_nodes() != len(node_features):
                    count_filter2 += 1
                    continue  # cleaning <10 graphs with this discrepancy

                g.edata["feat"] = edge_features
                g.ndata["feat"] = node_features

                self.graphs.append(g)
                self.label.append(F.tensor([molecule[3]]))
        self.label = F.tensor(self.label)
        print("Filtered graphs type 1/2: ", count_filter1, count_filter2)
        print("Filtered graphs: ", self.n_samples - len(self.graphs))

    def _process_zinc(self):
        self.graphs = []
        self.label = []
        for molecule in tqdm(self.dataset):

            node_features = molecule["atom_type"].long()

            adj = molecule["bond_type"]
            edge_list = (adj != 0).nonzero()  # converting adj matrix to edge_list

            edge_idxs_in_adj = edge_list.split(1, dim=1)
            edge_features = adj[edge_idxs_in_adj].reshape(-1).long()

            # Create the DGL Graph
            g = dgl.DGLGraph()
            g.add_nodes(molecule["num_atom"])
            g.ndata["feat"] = node_features

            for src, dst in edge_list:
                g.add_edges(src.item(), dst.item())
            g.edata["feat"] = edge_features

            self.graphs.append(g)
            self.label.append(molecule["logP_SA_cycle_normalized"])
        self.label = F.tensor(self.label)

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
            g = self.graphs[idx]
        else:
            g = self._transform(self.graphs[idx])
        return g, self.label[idx]

    def save(self):
        """save the graph list and the labels"""
        graph_path = os.path.join(self.save_path, "dgl_graph.bin")
        save_graphs(str(graph_path), self.graphs, {"labels": self.label})

    def load(self):
        graphs, label_dict = load_graphs(os.path.join(self.save_path, "dgl_graph.bin"))
        self.graphs = graphs
        self.label = label_dict["labels"]

    def get_idx_split(self):
        if self._name == "ZINC":
            test_split, val_split, train_split = 1000, 1000, 10000
        elif self._name == "ZINC-full":
            test_split, val_split, train_split = 5000, 24445, 220011
        elif self._name == "AqSol":
            test_split, val_split, train_split = 996, 996, 7831

        test_idx = np.arange(test_split)
        valid_idx = np.arange(val_split) + test_split
        train_idx = np.arange(train_split) + (test_split + val_split)

        assert self.label.shape[0] == test_split + val_split + train_split

        return {
            "train": F.tensor(train_idx, dtype=F.data_type_dict["int64"]),
            "valid": F.tensor(valid_idx, dtype=F.data_type_dict["int64"]),
            "test": F.tensor(test_idx, dtype=F.data_type_dict["int64"]),
        }

    def statistics(self):
        return None, 1, None

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.label.shape[0]

    @property
    def num_classes(self):
        """Number of labels for each graph, i.e. number of prediction tasks."""
        return 1

    @property
    def node_feat_size(self):
        return self.num_atom_type

    @property
    def edge_feat_size(self):
        return self.num_bond_type


def get_zinc(raw_dir=None, **kwargs):
    return MoleculeDatasetDGL(name="ZINC", raw_dir=raw_dir, **kwargs)


def get_zinc_full(raw_dir=None, **kwargs):
    return MoleculeDatasetDGL(name="ZINC-full", raw_dir=raw_dir, **kwargs)


def get_aqsol(raw_dir=None, **kwargs):
    return MoleculeDatasetDGL(name="AqSol", raw_dir=raw_dir, **kwargs)
