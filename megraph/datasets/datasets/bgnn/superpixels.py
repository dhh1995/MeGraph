# From https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/data/superpixels.py

import os
import pickle

import dgl
import numpy as np
from dgl import backend as F
from dgl.data.dgl_dataset import DGLDataset
from dgl.data.utils import download, load_graphs, save_graphs
from scipy.spatial.distance import cdist
from tqdm import tqdm


def sigma(dists, kth=8):
    # Compute sigma and reshape
    try:
        # Get k-nearest neighbors for each node
        knns = np.partition(dists, kth, axis=-1)[:, kth::-1]
        sigma = knns.sum(axis=1).reshape((knns.shape[0], 1)) / kth
    except ValueError:  # handling for graphs with num_nodes less than kth
        num_nodes = dists.shape[0]
        # this sigma value is irrelevant since not used for final compute_edge_list
        sigma = np.array([1] * num_nodes).reshape(num_nodes, 1)

    return sigma + 1e-8  # adding epsilon to avoid zero value of sigma


def compute_adjacency_matrix_images(coord, feat, use_feat=True, kth=8):
    coord = coord.reshape(-1, 2)
    # Compute coordinate distance
    c_dist = cdist(coord, coord)

    if use_feat:
        # Compute feature distance
        f_dist = cdist(feat, feat)
        # Compute adjacency
        A = np.exp(-((c_dist / sigma(c_dist)) ** 2) - (f_dist / sigma(f_dist)) ** 2)
    else:
        A = np.exp(-((c_dist / sigma(c_dist)) ** 2))

    # Convert to symmetric matrix
    A = 0.5 * (A + A.T)
    A[np.diag_indices_from(A)] = 0
    return A


def compute_edges_list(A, kth=8 + 1):
    # Get k-similar neighbor indices for each node

    num_nodes = A.shape[0]
    new_kth = num_nodes - kth

    if num_nodes > 9:
        knns = np.argpartition(A, new_kth - 1, axis=-1)[:, new_kth:-1]
        knn_values = np.partition(A, new_kth - 1, axis=-1)[:, new_kth:-1]  # NEW
    else:
        # handling for graphs with less than kth nodes
        # in such cases, the resulting graph will be fully connected
        knns = np.tile(np.arange(num_nodes), num_nodes).reshape(num_nodes, num_nodes)
        knn_values = A  # NEW

        # removing self loop
        if num_nodes != 1:
            knn_values = A[knns != np.arange(num_nodes)[:, None]].reshape(
                num_nodes, -1
            )  # NEW
            knns = knns[knns != np.arange(num_nodes)[:, None]].reshape(num_nodes, -1)
    return knns, knn_values  # NEW


class SuperPixDGL(DGLDataset):
    _url = "https://www.dropbox.com/s/y2qwa77a0fxem47/superpixels.zip?dl=1"

    def __init__(
        self,
        name,
        raw_dir=None,
        force_reload=False,
        verbose=False,
        transform=None,
        use_mean_px=False,
        use_coord=True,
    ):
        assert name in ["CIFAR10", "MNIST"]
        if name == "MNIST":
            self.img_size = 28
        elif name == "CIFAR10":
            self.img_size = 32
        self.use_mean_px = use_mean_px
        self.use_coord = use_coord
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
        file_path = os.path.join(self.raw_dir, "superpixels.zip")
        if not os.path.exists(file_path):
            download_cmd = f"curl {self._url} -o {file_path} -J -L -k"
            os.system(download_cmd)
        if not os.path.exists(os.path.join(self.raw_dir, "superpixels")):
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
        if self._name == "MNIST":
            test_file = open(
                os.path.join(self.raw_dir, "superpixels/mnist_75sp_%s.pkl" % "test"),
                "rb",
            )
            self.test_labels, self.test_data = pickle.load(test_file)
            self.test_labels = F.tensor(
                self.test_labels, dtype=F.data_type_dict["int64"]
            )

            train_file = open(
                os.path.join(self.raw_dir, "superpixels/mnist_75sp_%s.pkl" % "train"),
                "rb",
            )
            self.train_labels, self.train_data = pickle.load(train_file)
            self.train_labels = F.tensor(
                self.train_labels, dtype=F.data_type_dict["int64"]
            )

        elif self._name == "CIFAR10":
            test_file = open(
                os.path.join(self.raw_dir, "superpixels/cifar10_150sp_%s.pkl" % "test"),
                "rb",
            )
            self.test_labels, self.test_data = pickle.load(test_file)
            self.test_labels = F.tensor(
                self.test_labels, dtype=F.data_type_dict["int64"]
            )

            train_file = open(
                os.path.join(
                    self.raw_dir, "superpixels/cifar10_150sp_%s.pkl" % "train"
                ),
                "rb",
            )
            self.train_labels, self.train_data = pickle.load(train_file)
            self.train_labels = F.tensor(
                self.train_labels, dtype=F.data_type_dict["int64"]
            )

        self.data = self.test_data + self.train_data
        self.label = F.cat([self.test_labels, self.train_labels], dim=0)
        self.n_samples = len(self.test_labels) + len(self.train_labels)

    def process(self):
        self.prepare()
        print(f"processing {self.n_samples} graphs")
        self.Adj_matrices, self.node_features, self.edges_lists, self.edge_features = (
            [],
            [],
            [],
            [],
        )
        for index, sample in tqdm(enumerate(self.data)):
            mean_px, coord = sample[:2]

            try:
                coord = coord / self.img_size
            except AttributeError:
                VOC_has_variable_image_sizes = True

            if self.use_mean_px:
                A = compute_adjacency_matrix_images(
                    coord, mean_px
                )  # using super-pixel locations + features
            else:
                A = compute_adjacency_matrix_images(
                    coord, mean_px, False
                )  # using only super-pixel locations
            edges_list, edge_values_list = compute_edges_list(A)  # NEW

            N_nodes = A.shape[0]

            mean_px = mean_px.reshape(N_nodes, -1)
            coord = coord.reshape(N_nodes, 2)
            x = np.concatenate((mean_px, coord), axis=1)

            edge_values_list = edge_values_list.reshape(-1)  # NEW # TO DOUBLE-CHECK !

            self.node_features.append(x)
            self.edge_features.append(edge_values_list)  # NEW
            self.Adj_matrices.append(A)
            self.edges_lists.append(edges_list)

        self.graphs = []
        for index in tqdm(range(len(self.data))):
            g = dgl.DGLGraph()
            g.add_nodes(self.node_features[index].shape[0])
            g.ndata["feat"] = F.tensor(self.node_features[index]).half()

            for src, dsts in enumerate(self.edges_lists[index]):
                # handling for 1 node where the self loop would be the only edge
                # since, VOC Superpixels has few samples (5 samples) with only 1 node
                if self.node_features[index].shape[0] == 1:
                    g.add_edges(src, dsts)
                else:
                    g.add_edges(src, dsts[dsts != src])

            # adding edge features for Residual Gated ConvNet
            edge_feat_dim = g.ndata["feat"].shape[1]  # dim same as node feature dim
            # g.edata['feat'] = torch.ones(g.number_of_edges(), edge_feat_dim).half()
            g.edata["feat"] = (
                F.tensor(self.edge_features[index]).unsqueeze(1).half()
            )  # NEW

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
        if self._name == "CIFAR10":
            test_split, val_split, train_split = 10000, 5000, 45000
        elif self._name == "MNIST":
            test_split, val_split, train_split = 10000, 5000, 55000

        test_idx = np.arange(test_split)
        valid_idx = np.arange(val_split) + test_split
        train_idx = np.arange(train_split) + (test_split + val_split)

        return {
            "train": F.tensor(train_idx, dtype=F.data_type_dict["int64"]),
            "valid": F.tensor(valid_idx, dtype=F.data_type_dict["int64"]),
            "test": F.tensor(test_idx, dtype=F.data_type_dict["int64"]),
        }

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.label.shape[0]

    def statistics(self):
        return None, 10, None

    @property
    def num_classes(self):
        """Number of labels for each graph, i.e. number of prediction tasks."""
        return 10


def get_cifar10(raw_dir=None, **kwargs):
    return SuperPixDGL(name="CIFAR10", raw_dir=raw_dir, **kwargs)


def get_mnist(raw_dir=None, **kwargs):
    return SuperPixDGL(name="MNIST", raw_dir=raw_dir, **kwargs)
