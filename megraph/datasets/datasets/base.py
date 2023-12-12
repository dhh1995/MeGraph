#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : base.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
#
# Distributed under terms of the MIT license.

import os.path as osp
import pickle

import dgl
import torch
from dgl import DGLGraph
from dgl import backend as F
from dgl.data.dgl_dataset import DGLDataset
from dgl.data.utils import load_graphs, save_graphs
from dgl.transforms import reorder_graph
from megraph.rng_utils import sample_between_min_max
from networkx import DiGraph
from tqdm import tqdm


class BaseSyntheticDataset(DGLDataset):
    def __init__(
        self,
        name,
        url=None,
        raw_dir=None,
        force_reload=False,
        verbose=False,
        transform=None,
    ):
        self._graphs = []
        self._labels = []
        super(BaseSyntheticDataset, self).__init__(
            name=name,
            url=url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )

    def build_graph_from_nx(
        self,
        graph: DiGraph,
        node_feat=None,
        edge_feat=None,
        node_label=None,
        edge_label=None,
    ) -> DGLGraph:
        # set edge attrs to networkx graph, convert to directed, then to dgl graph
        edge_attrs = []
        if edge_feat is not None:
            edge_attrs.append("feat")
        if edge_label is not None:
            edge_attrs.append("label")
        if len(edge_attrs) == 0:
            edge_attrs = None
        for i, (u, v) in enumerate(graph.edges()):
            if edge_feat is not None:
                graph[u][v]["feat"] = edge_feat[i]
            if edge_label is not None:
                graph[u][v]["label"] = edge_label[i]
        if edge_attrs is not None and not graph.is_directed():
            graph = graph.to_directed()

        graph = dgl.from_networkx(graph, edge_attrs=edge_attrs)
        if node_feat is None:
            node_feat = F.ones((graph.num_nodes(), 1), F.float32, F.cpu())
        else:
            node_feat = F.tensor(node_feat, F.float32)
        graph.ndata["feat"] = node_feat
        if node_label is not None:
            graph.ndata["label"] = F.tensor(node_label, F.float32)
            # store in float no matter float or int
            # convert to int64 when getting data depending on the task
        graph = reorder_graph(
            graph, node_permute_algo="rcmk", edge_permute_algo="dst", store_ids=True
        )
        return graph

    def process(self):
        raise NotImplementedError()

    @property
    def graph_path(self):
        return osp.join(self.save_path, f"{self.name}_dgl_graph.bin")

    @property
    def label_path(self):
        return osp.join(self.save_path, f"{self.name}_dgl_label.pkl")

    def save(self):
        save_graphs(str(self.graph_path), self._graphs, {"labels": self._labels})

    def has_cache(self):
        return osp.exists(self.graph_path)

    def load(self):
        self._graphs, label_dict = load_graphs(str(self.graph_path))
        self._labels = label_dict["labels"]

    def __getitem__(self, idx):
        if len(self) == 1:
            assert idx == 0, "This dataset has only one graph."
        graph = self._graphs[idx]
        if self._transform is not None:
            graph = self._transform(graph)
        ret = graph
        if len(self._labels) > 0:
            ret = (graph, self._labels[idx])
        return ret

    def _get_num_tasks(self):
        raise NotImplementedError()

    def _get_num_classes(self):
        raise NotImplementedError()

    def _get_node_feat_size(self):
        raise NotImplementedError()

    def _get_edge_feat_size(self):
        return 0

    @property
    def num_tasks(self):
        return self._get_num_tasks()

    @property
    def num_classes(self):
        return self._get_num_classes()

    @property
    def node_feat_size(self):
        return self._get_node_feat_size()

    @property
    def edge_feat_size(self):
        return self._get_edge_feat_size()

    def __len__(self):
        return len(self._graphs)


class SyntheticMultiGraphDataset(BaseSyntheticDataset):
    def __init__(
        self,
        name,
        n_graphs,
        num_nodes_min_max=[10, 20],
        train_ratio=0.8,
        test_ratio=0.1,
        **kwargs,
    ):
        self.n_graphs = n_graphs
        self.num_nodes_min_max = num_nodes_min_max
        num_train = int(n_graphs * train_ratio)
        num_test = int(n_graphs * test_ratio)
        idx = torch.arange(self.n_graphs)
        self.train_idx = idx[:num_train]
        self.val_idx = idx[num_train:-num_test]
        self.test_idx = idx[-num_test:]
        super(SyntheticMultiGraphDataset, self).__init__(name=name, **kwargs)

    def _get_graph(self, n):
        raise NotImplementedError()

    def get_graph(self):
        while True:
            num_nodes = sample_between_min_max(self.num_nodes_min_max)
            result = self._get_graph(num_nodes)
            if result is not None:
                return result

    def process(self):
        self._graphs = []
        self._labels = []
        print("generating graphs and labels")
        for i in tqdm(range(self.n_graphs)):
            graph, res = self.get_graph()
            feat = res["feat"]
            efeat = res.get("efeat", None)
            nlabel = res.get("nlabel", None)
            elabel = res.get("elabel", None)
            glabel = res.get("glabel", None)
            graph = self.build_graph_from_nx(
                graph,
                node_feat=feat,
                edge_feat=efeat,
                node_label=nlabel,
                edge_label=elabel,
            )
            self._graphs.append(graph)
            if glabel is not None:
                self._labels.append(glabel)
        self._labels = F.tensor(self._labels, F.float32)
        # store in float no matter float or int
        # convert to int64 when getting data depending on the task
        if len(self._labels) > 0:
            for split, idx in zip(
                ["train", "val", "test"], [self.train_idx, self.val_idx, self.test_idx]
            ):
                print(f"{split} label mean: {self._labels[idx].mean()}")
