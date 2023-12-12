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


class PeptidesStructuralDataset(DGLDataset):
    def __init__(
        self,
        name="peptides_structure",
        raw_dir=None,
        force_reload=False,
        verbose=False,
        transform=None,
        smiles2graph=smiles2graph,
    ):
        """
        PyG dataset of 15,535 small peptides represented as their molecular
        graph (SMILES) with 11 regression targets derived from the peptide's
        3D structure.

        The original amino acid sequence representation is provided in
        'peptide_seq' and the distance between atoms in 'self_dist_matrix' field
        of the dataset file, but not used here as any part of the input.

        The 11 regression targets were precomputed from molecule XYZ:
            Inertia_mass_[a-c]: The principal component of the inertia of the
                mass, with some normalizations. Sorted
            Inertia_valence_[a-c]: The principal component of the inertia of the
                Hydrogen atoms. This is basically a measure of the 3D
                distribution of hydrogens. Sorted
            length_[a-c]: The length around the 3 main geometric axis of
                the 3D objects (without considering atom types). Sorted
            Spherocity: SpherocityIndex descriptor computed by
                rdkit.Chem.rdMolDescriptors.CalcSpherocityIndex
            Plane_best_fit: Plane of best fit (PBF) descriptor computed by
                rdkit.Chem.rdMolDescriptors.CalcPBF
        Args:
            root (string): Root directory where the dataset should be saved.
            smiles2graph (callable): A callable function that converts a SMILES
                string into a graph object. We use the OGB featurization.
                * The default smiles2graph requires rdkit to be installed *
        """

        self.smiles2graph = smiles2graph

        self._url = "https://www.dropbox.com/s/464u3303eu2u4zp/peptide_structure_dataset.csv.gz?dl=1"
        self.version = (
            "9786061a34298a0684150f2e4ff13f47"  # MD5 hash of the intended dataset file
        )
        self._url_stratified_split = "https://www.dropbox.com/s/9dfifzft1hqgow6/splits_random_stratified_peptide_structure.pickle?dl=1"
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
        if decide_download(self.url):
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
            os.path.join(self.raw_path, "peptide_structure_dataset.csv.gz")
        )
        smiles_list = data_df["smiles"]
        target_names = [
            "Inertia_mass_a",
            "Inertia_mass_b",
            "Inertia_mass_c",
            "Inertia_valence_a",
            "Inertia_valence_b",
            "Inertia_valence_c",
            "length_a",
            "length_b",
            "length_c",
            "Spherocity",
            "Plane_best_fit",
        ]
        # Normalize to zero mean and unit standard deviation.
        data_df.loc[:, target_names] = data_df.loc[:, target_names].apply(
            lambda x: (x - x.mean()) / x.std(), axis=0
        )

        print("Converting SMILES strings into graphs...")
        self.graphs, self.label = [], []
        for i in tqdm(range(len(smiles_list))):
            smiles = smiles_list[i]
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

            label = data_df.iloc[i][target_names]
            # label = F.tensor([label], dtype=F.data_type_dict["int64"])
            label = np.array(label, dtype=np.float32)

            self.graphs.append(g)
            self.label.append(label)

        self.label = F.tensor(np.array(self.label), dtype=F.data_type_dict["float32"])

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
        return None, 11, None

    @property
    def num_tasks(self):
        """Number of labels for each graph, i.e. number of prediction tasks."""
        return 11

    def get_idx_split(self):
        """Get dataset splits.

        Returns:
            Dict with 'train', 'val', 'test', splits indices.
        """
        split_file = os.path.join(
            self.raw_path, "splits_random_stratified_peptide_structure.pickle"
        )
        with open(split_file, "rb") as f:
            splits = pickle.load(f)
        split_dict = replace_numpy_with_torchtensor(splits)
        split_dict["valid"] = split_dict["val"]
        split_dict.pop("val")
        return split_dict


if __name__ == "__main__":
    dataset = PeptidesStructuralDataset()
    print(dataset)
    print(dataset.data.edge_index)
    print(dataset.data.edge_index.shape)
    print(dataset.data.x.shape)
    print(dataset[100])
    print(dataset[100].y)
    print(dataset.get_idx_split())
