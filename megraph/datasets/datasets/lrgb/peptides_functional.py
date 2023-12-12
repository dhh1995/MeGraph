import hashlib
import os
import pickle
import shutil

import dgl
import numpy as np
import pandas as pd
from dgl import backend as F
from dgl.data.dgl_dataset import DGLDataset
from dgl.data.utils import download, load_graphs, save_graphs
from ogb.utils import smiles2graph
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download
from torch_geometric.data import download_url
from tqdm import tqdm


class PeptidesFunctionalDataset(DGLDataset):
    def __init__(
        self,
        name="peptides_functional",
        raw_dir=None,
        force_reload=False,
        verbose=False,
        transform=None,
        smiles2graph=smiles2graph,
    ):
        """
        PyG dataset of 15,535 peptides represented as their molecular graph
        (SMILES) with 10-way multi-task binary classification of their
        functional classes.

        The goal is use the molecular representation of peptides instead
        of amino acid sequence representation ('peptide_seq' field in the file,
        provided for possible baseline benchmarking but not used here) to test
        GNNs' representation capability.

        The 10 classes represent the following functional classes (in order):
            ['antifungal', 'cell_cell_communication', 'anticancer',
            'drug_delivery_vehicle', 'antimicrobial', 'antiviral',
            'antihypertensive', 'antibacterial', 'antiparasitic', 'toxic']

        Args:
            root (string): Root directory where the dataset should be saved.
            smiles2graph (callable): A callable function that converts a SMILES
                string into a graph object. We use the OGB featurization.
                * The default smiles2graph requires rdkit to be installed *
        """

        self.smiles2graph = smiles2graph
        self._url = "https://www.dropbox.com/s/ol2v01usvaxbsr8/peptide_multi_class_dataset.csv.gz?dl=1"
        self.version = (
            "701eb743e899f4d793f0e13c8fa5a1b4"  # MD5 hash of the intended dataset file
        )
        self._url_stratified_split = "https://www.dropbox.com/s/j4zcnx2eipuo0xz/splits_random_stratified_peptide.pickle?dl=1"
        self.md5sum_stratified_split = "5a0114bdadc80b94fc7ae974f13ef061"

        super().__init__(
            name=name,
            url=self._url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )
        # Check version and update if necessary.
        release_tag = os.path.join(self.raw_path, self.version)
        if os.path.isdir(self.raw_path) and (not os.path.exists(release_tag)):
            print(f"{self.__class__.__name__} has been updated.")
            if input("Will you update the dataset now? (y/N)\n").lower() == "y":
                shutil.rmtree(self.raw_path)

    def has_cache(self):
        graph_path = os.path.join(self.save_path, "dgl_graph.bin")
        return os.path.exists(graph_path)

    def _md5sum(self, path):
        hash_md5 = hashlib.md5()
        with open(path, "rb") as f:
            buffer = f.read()
            hash_md5.update(buffer)
        return hash_md5.hexdigest()

    def download(self):
        if decide_download(self._url):
            path = download_url(self._url, self.raw_path)
            # Save to disk the MD5 hash of the downloaded file.
            hash = self._md5sum(path)
            if hash != self.version:
                raise ValueError("Unexpected MD5 hash of the downloaded file")
            open(os.path.join(self.raw_path, hash), "w").close()
            # Download train/val/test splits.
            path_split1 = download_url(self._url_stratified_split, self.raw_path)
            assert self._md5sum(path_split1) == self.md5sum_stratified_split
        else:
            print("Stop download.")
            exit(-1)

    def process(self):
        data_df = pd.read_csv(
            os.path.join(self.raw_path, "peptide_multi_class_dataset.csv.gz")
        )
        data = data_df["smiles"]
        label = data_df["labels"].to_numpy()
        self.n_samples = len(label)

        print(f"processing {self.n_samples} graphs")
        print("Converting SMILES strings into graphs...")

        self.graphs, self.label = [], []
        for i in tqdm(range(self.n_samples)):
            smiles = data[i]
            graph = self.smiles2graph(smiles)

            assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
            assert len(graph["node_feat"]) == graph["num_nodes"]

            g = dgl.DGLGraph()

            g.add_nodes(int(graph["num_nodes"]))
            g.ndata["feat"] = F.tensor(
                graph["node_feat"], dtype=F.data_type_dict["int64"]
            )

            edge_index = graph["edge_index"]
            g.add_edges(edge_index[0], edge_index[1])
            g.edata["feat"] = F.tensor(
                graph["edge_feat"], dtype=F.data_type_dict["int64"]
            )

            # g.ndata["label"] = F.tensor(label, dtype=F.data_type_dict["int64"])
            self.graphs.append(g)
            self.label.append(np.array(eval(label[i])))

        self.label = F.tensor(np.array(self.label), dtype=F.data_type_dict["int64"])

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

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.label.shape[0]

    def statistics(self):
        return None, 10, None

    @property
    def num_classes(self):
        """Number of labels for each graph, i.e. number of prediction tasks."""
        return 10

    def get_idx_split(self):
        """Get dataset splits.

        Returns:
            Dict with 'train', 'val', 'test', splits indices.
        """
        split_file = os.path.join(
            self.raw_path, "splits_random_stratified_peptide.pickle"
        )
        with open(split_file, "rb") as f:
            splits = pickle.load(f)
        split_dict = replace_numpy_with_torchtensor(splits)
        split_dict["valid"] = split_dict["val"]
        split_dict.pop("val")
        return split_dict
