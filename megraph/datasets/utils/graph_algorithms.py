#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : graph.py
# Author : Honghua Dong, Yang Yu
# Email  : dhh19951@gmail.com, 773964676@qq.com
#
# Distributed under terms of the MIT license.

from typing import List

import networkx as nx
import numpy as np

__all__ = [
    "get_diameter_length_and_nodes",
    "get_disjoint_components_size",
    "check_is_bipartite_graph",
]


def get_diameter_length_and_nodes(g: nx.DiGraph, need_nodes_label=True):
    """Calculate the diameter of the given graph. Also include the nodes on any
    diameter when needed."""
    d = nx.floyd_warshall_numpy(g)
    diameter = d.max()
    if not need_nodes_label:
        return diameter

    n = g.number_of_nodes()
    nodes_label = np.zeros(shape=(n,))
    source, target = np.where(d == diameter)
    for i in range(n):
        for s, t in zip(source, target):
            if d[s, i] + d[i, t] == diameter:
                nodes_label[i] = 1
                break

    return diameter, nodes_label


def find_root(f, x):
    """Find the root of the tree that x belongs to."""
    if f[x] == x:
        return x
    fx = find_root(f, f[x])
    f[x] = fx  # Shortcut
    return fx


def get_disjoint_components_size(g: nx.Graph, selected=None) -> List[int]:
    """Get the size of disjoint components in the (selected) graph nodes using
    Disjoint-set data structure."""

    num = n = g.number_of_nodes()
    f = np.arange(n)
    cnt = np.ones(n)
    for e in g.edges():
        x, y = e
        if selected is not None:
            if not selected[x] or not selected[y]:
                continue
        x = find_root(f, x)
        y = find_root(f, y)
        if x != y:
            if np.random.randint(2) == 0:
                x, y = y, x
            f[x] = y
            cnt[y] = cnt[y] + cnt[x]
            num -= 1
    return cnt[np.where(f == np.arange(n))]


def check_is_bipartite_graph(g: nx.Graph, selected_edges=None):
    if selected_edges is not None:
        edges = np.array(g.edges())[selected_edges]
        g = nx.Graph()
        g.add_edges_from(edges)
    return int(nx.is_bipartite(g))


def count_connected_components(g: nx.Graph, selected_edges=None):
    if selected_edges is not None:
        n = g.number_of_nodes()
        edges = np.array(g.edges())[selected_edges]
        g = nx.Graph()
        g.add_nodes_from(list(range(n)))
        g.add_edges_from(edges)
    return len(list(nx.connected_components(g)))
