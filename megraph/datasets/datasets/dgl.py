#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : dgl.py
# Author : Jiawei Xu, Honghua Dong
# Email  : 1138217508@qq.com, dhh19951@gmail.com
#
# Distributed under terms of the MIT license.


"""The ``dgl.data`` package contains datasets hosted by DGL and also utilities
for downloading, processing, saving and loading data from external resources.
"""

from __future__ import absolute_import

import functools
import os.path as osp

import numpy as np
import torch
from dgl.data import BA2MotifDataset
from dgl.data.bitcoinotc import BitcoinOTCDataset

# Node Prediction Datasets
from dgl.data.citation_graph import (
    CiteseerGraphDataset,
    CoraGraphDataset,
    PubmedGraphDataset,
)
from dgl.data.fakenews import FakeNewsDataset
from dgl.data.gdelt import GDELTDataset
from dgl.data.gindt import GINDataset
from dgl.data.gnn_benchmark import (
    AmazonCoBuyComputerDataset,
    AmazonCoBuyPhotoDataset,
    CoauthorCSDataset,
    CoauthorPhysicsDataset,
)
from dgl.data.icews18 import ICEWS18Dataset

# Edge Prediction Datasets
from dgl.data.knowledge_graph import FB15k237Dataset, FB15kDataset, WN18Dataset
from dgl.data.minigc import MiniGCDataset

# Graph Prediction Datasets
from dgl.data.qm7b import QM7bDataset
from dgl.data.qm9 import QM9Dataset
from dgl.data.qm9_edge import QM9EdgeDataset
from dgl.data.reddit import RedditDataset
from dgl.data.synthetic import (
    BACommunityDataset,
    BAShapeDataset,
    TreeCycleDataset,
    TreeGridDataset,
)
from dgl.data.tu import LegacyTUDataset

# Datasets utils
# from dgl.data.utils import *
from dgl.data.utils import load_graphs, load_info, save_graphs, save_info
from dgl.transforms import RowFeatNormalizer

from . import graph_dataset_manager as manager


def minigc(raw_dir, **kwargs):
    # min_nv, max_nv = 20, 50
    min_nv, max_nv = 100, 200
    return MiniGCDataset(num_graphs=500, min_num_v=min_nv, max_num_v=max_nv, **kwargs)


class CustomTuDataset(LegacyTUDataset):
    """Extended Tu dataset that include node label into node feat"""

    def process(self):
        super().process()

        DS_indicator = self._idx_from_zero(
            np.genfromtxt(self._file_path("graph_indicator"), dtype=int)
        )
        node_idx_list = []
        self.max_num_node = 0
        for idx in range(np.max(DS_indicator) + 1):
            node_idx = np.where(DS_indicator == idx)
            node_idx_list.append(node_idx[0])
            if len(node_idx[0]) > self.max_num_node:
                self.max_num_node = len(node_idx[0])

        if self.data_mode != "node_label":
            try:
                DS_node_labels = self._idx_from_zero(
                    np.loadtxt(self._file_path("node_labels"), dtype=int)
                )
                one_hot_node_labels = self._to_onehot(DS_node_labels)
                for idxs, g in zip(node_idx_list, self.graph_lists):
                    node_label = torch.tensor(
                        one_hot_node_labels[idxs, :], dtype=torch.float32
                    )
                    g.ndata["feat"] = torch.cat([g.ndata["feat"], node_label], dim=-1)
                # self.data_mode = self.data_mode + "_node_label"
            except IOError:
                print("No Node Label Data")

    def save(self):
        graph_path = osp.join(
            self.save_path, "custom_tu_{}_{}.bin".format(self.name, self.hash)
        )
        info_path = osp.join(
            self.save_path, "custom_tu_{}_{}.pkl".format(self.name, self.hash)
        )
        label_dict = {"labels": self.graph_labels}
        info_dict = {"max_num_node": self.max_num_node, "num_labels": self.num_labels}
        save_graphs(str(graph_path), self.graph_lists, label_dict)
        save_info(str(info_path), info_dict)

    def load(self):
        graph_path = osp.join(
            self.save_path, "custom_tu_{}_{}.bin".format(self.name, self.hash)
        )
        info_path = osp.join(
            self.save_path, "custom_tu_{}_{}.pkl".format(self.name, self.hash)
        )
        graphs, label_dict = load_graphs(str(graph_path))
        info_dict = load_info(str(info_path))

        self.graph_lists = graphs
        self.graph_labels = label_dict["labels"]
        self.max_num_node = info_dict["max_num_node"]
        self.num_labels = info_dict["num_labels"]

    def has_cache(self):
        graph_path = osp.join(
            self.save_path, "custom_tu_{}_{}.bin".format(self.name, self.hash)
        )
        info_path = osp.join(
            self.save_path, "custom_tu_{}_{}.pkl".format(self.name, self.hash)
        )
        if osp.exists(graph_path) and osp.exists(info_path):
            return True
        return False


GNN_BENCHMARK_DATASETS = {
    "coauthorcs": CoauthorCSDataset,
    "coauthorphysics": CoauthorPhysicsDataset,
    "amazoncobuyphoto": AmazonCoBuyPhotoDataset,
    "amazoncobuycomputer": AmazonCoBuyComputerDataset,
}
INDUCTIVE_DATASETS = {"ppi"}
NODE_PREDICTION_DATASETS = {
    "cora": CoraGraphDataset,
    "citeseer": CiteseerGraphDataset,
    "pubmed": PubmedGraphDataset,
    "reddit": RedditDataset,  # Too large, need subgraph training
    # synthetic
    "bashape": BAShapeDataset,
    "bacommunity": BACommunityDataset,
    "treegrid": TreeGridDataset,
    "treecycle": TreeCycleDataset,
}
LINK_PREDICTION_DATASETS = {
    "fb15k": FB15kDataset,
    "fb15k237": FB15k237Dataset,
    "wn18": WN18Dataset,
    "bitcoinotc": BitcoinOTCDataset,
    "icews18": ICEWS18Dataset,
    "gdelt": GDELTDataset,
}
GRAPH_PREDICTION_DATASETS = {
    "qm7b": QM7bDataset,
    "qm9": QM9Dataset,
    "qm9_edge": QM9EdgeDataset,
    "minigc": minigc,
    "fakenews": FakeNewsDataset,
    "ba2motif": BA2MotifDataset,
}
COLLECTION_DATASETS = {
    # "tu": TUDataset,
    "tu": LegacyTUDataset,
    "tuc": CustomTuDataset,
    "gin": GINDataset,
}


def get_gnn_benchmark(name, transform=RowFeatNormalizer(subtract_min=True)):
    dataset = GNN_BENCHMARK_DATASETS[name](transform=transform)
    return dataset


for k, v in GNN_BENCHMARK_DATASETS.items():
    NODE_PREDICTION_DATASETS[k] = functools.partial(get_gnn_benchmark, name=k)
for k, v in NODE_PREDICTION_DATASETS.items():
    inductive = k in INDUCTIVE_DATASETS
    loss_func = "BCE" if k in ["ppi"] else "CE"
    manager.add_dataset(
        k, dict(fn=v, task="npred", inductive=inductive, loss_func=loss_func)
    )
for k, v in LINK_PREDICTION_DATASETS.items():
    manager.add_dataset(k, dict(fn=v, task="lpred", inductive=True))
for k, v in GRAPH_PREDICTION_DATASETS.items():
    manager.add_dataset(k, dict(fn=v, task="gpred", inductive=True))
for k, v in COLLECTION_DATASETS.items():
    manager.add_dataset(k, dict(fn=v, task="gpred", inductive=True, need_subname=True))
