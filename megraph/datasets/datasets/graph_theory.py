#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : graph_theory.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
#
# Distributed under terms of the MIT license.

"""Support a set of graph theory tasks"""

from functools import partial

import numpy as np
from megraph.rng_utils import sample_between_min_max

from .base import SyntheticMultiGraphDataset


class GraphTheoryDataset(SyntheticMultiGraphDataset):
    def __init__(
        self,
        name,
        graph_builder,
        feat_label_computer,
        n_graphs=500,
        # graph builder args
        num_nodes_min_max=[10, 20],
        degree_min_max=None,
        # task args
        num_classes=1,
        num_tasks=1,
        feat_size=1,
        edge_feat_size=0,
        **kwargs,
    ):
        self.graph_builder = graph_builder
        self.feat_label_computer = feat_label_computer
        self._degree_min_max = degree_min_max
        self._num_classes = num_classes
        self._num_tasks = num_tasks
        self._feat_size = feat_size
        self._edge_feat_size = edge_feat_size
        super(GraphTheoryDataset, self).__init__(
            name=name,
            n_graphs=n_graphs,
            num_nodes_min_max=num_nodes_min_max,
            **kwargs,
        )

    def _get_graph(self, n):
        degree = None
        if self._degree_min_max is not None:
            degree = sample_between_min_max(self._degree_min_max)
        graph = self.graph_builder(n, degree=degree)
        res = self.feat_label_computer(graph, self._feat_size)
        if res is None:
            return None
        if "feat" not in res:  # Default node feat (all one)
            res["feat"] = np.ones((n, self._feat_size))
        return graph, res

    def _get_num_classes(self):
        return self._num_classes

    def _get_num_tasks(self):
        return self._num_tasks

    def _get_node_feat_size(self):
        return self._feat_size

    def _get_edge_feat_size(self):
        return self._edge_feat_size
