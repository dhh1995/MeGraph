#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : pseudoforest.py
# Author : Honghua Dong, Yang Yu
# Email  : dhh19951@gmail.com, 773964676@qq.com
#
# Distributed under terms of the MIT license.

import dgl
import numpy as np
from dgl import backend as F
from dgl.transforms import reorder_graph

from ..utils import generate_pseudotree
from . import BaseSyntheticDataset, register_function


class BasePseudoTreeDataset(BaseSyntheticDataset):
    def __init__(
        self,
        name,
        tree_size_min_max=[5, 50],
        **kwargs,
    ):
        self.tree_size_min_max = tree_size_min_max
        super(BasePseudoTreeDataset, self).__init__(name=name, **kwargs)

    def generate_graph(self, n_nodes):
        graph, node_label = generate_pseudotree(n_nodes)
        graph = self.build_graph_from_nx(graph, node_label=node_label)
        return graph

    def _get_num_classes(self):
        return 2

    def _get_node_feat_size(self):
        return 1


@register_function("pseudoforest", dict(inductive=False, task="npred"))
class PseudoForestDataset(BasePseudoTreeDataset):
    def __init__(self, n_nodes=5000, **kwargs):
        self.n_nodes = n_nodes
        super(PseudoForestDataset, self).__init__(name="pseudoforest", **kwargs)

    def process(self):
        mi, ma = self.tree_size_min_max

        remain_n_nodes = self.n_nodes
        graphs = []
        while remain_n_nodes > 0:
            n_nodes = remain_n_nodes
            if n_nodes > ma:
                n_nodes = np.random.randint(ma - mi + 1) + mi
            remain_n_nodes -= n_nodes
            graph = self.generate_graph(n_nodes)
            graphs.append(graph)
        self._graphs = [dgl.batch(graphs)]


@register_function("pseudotree", dict(inductive=True, task="npred", need_adapter=False))
class PseudoTreeDataset(BasePseudoTreeDataset):
    def __init__(self, n_graphs=500, **kwargs):
        self.n_graphs = n_graphs
        super(PseudoTreeDataset, self).__init__(name="pseudotree", **kwargs)

    def process(self):
        self._graphs = []
        for _ in range(self.n_graphs):
            mi, ma = self.tree_size_min_max
            n_nodes = np.random.randint(ma - mi + 1) + mi
            graph = self.generate_graph(n_nodes)
            self._graphs.append(graph)
