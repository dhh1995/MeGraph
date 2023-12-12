#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : graph_generators.py
# Author : Honghua Dong, Yang Yu
# Email  : dhh19951@gmail.com, 773964676@qq.com
#
# Distributed under terms of the MIT license.

from collections import deque
from enum import Enum
from functools import partial
from typing import List, Tuple

import networkx as nx
import numpy as np
import numpy.random as random
from megraph.datasets.utils.graph_generation import (barabasi_albert,
                                                     caterpillar, caveman,
                                                     erdos_renyi,
                                                     generate_graph_geo,
                                                     generate_graph_sbm, grid,
                                                     ladder, line, lobster,
                                                     star, tree)
from megraph.rng_utils import (sample_between_min_max, sample_from_mixture,
                               sample_partition)

__all__ = [
    "generate_graph_pseudotree",
    "generate_graph_cycle",
    "get_random_graph_builder",
    "generate_pseudotree",
]


def sample_random_edge(g: nx.Graph):
    n = g.number_of_nodes()
    while True:
        u, v = random.randint(n), random.randint(n)
        if (not g.has_edge(u, v)) and (u != v):
            return u, v


def generate_graph_pseudotree(
    num_nodes: int,
    cycle_ratio_min_max: List[float] = [0.3, 0.6],
    partition_method: str = "sep",
) -> Tuple[nx.DiGraph, int]:
    """[v2] Generate a random tree with sampled cycle length"""
    cycle_ratio = sample_between_min_max(cycle_ratio_min_max)
    cycle_len = max(min(3, num_nodes), int(num_nodes * cycle_ratio))
    g = nx.cycle_graph(cycle_len)
    expander_sizes = sample_partition(
        num_nodes - cycle_len, cycle_len, method=partition_method
    )
    cur_idx = cycle_len
    for i in range(cycle_len):
        tree_size = expander_sizes[i] + 1  # the root
        if tree_size > 1:
            tree = nx.random_tree(tree_size)
            # Merge tree to g while the root of the tree is node i on g
            re_index = lambda x: i if x == 0 else cur_idx + x - 1
            for u, v in tree.edges():
                g.add_edge(re_index(u), re_index(v))
            cur_idx += tree_size - 1
    return g, cycle_len


def generate_graph_cycle(n: int) -> nx.DiGraph:
    return nx.cycle_graph(n)


def generate_graph_blooming(n: int, degree=None, edge_factor=0.2) -> nx.DiGraph:
    """A fractal tree plus some random edges"""
    degree = degree or 2
    g = nx.empty_graph(n)
    edges = []
    cur = 1
    q = deque([0])
    while cur < n:
        x = q.popleft()
        for _ in range(degree):
            if cur < n:
                edges.append((x, cur))
                q.append(cur)
                cur += 1
    g.add_edges_from(edges)
    # random new edges
    for _ in range(int(n * edge_factor)):
        u, v = sample_random_edge(g)
        g.add_edge(u, v)
    return g


# Graph generators and default graph scales
GRAPH_GENERATORS_PAIRS = [
    ("er", erdos_renyi),
    ("ba", barabasi_albert),
    ("grid", grid),
    ("caveman", caveman),
    ("tree", tree),
    ("ladder", ladder),
    ("line", line),
    ("star", star),
    ("caterpillar", caterpillar),
    ("lobster", lobster),
    ("cycle", generate_graph_cycle),
    ("pseudotree", generate_graph_pseudotree),
    ("geo", generate_graph_geo),
    ("bloom", generate_graph_blooming),
    ("sbm", generate_graph_sbm),
]
GRAPH_GENERATOR_NAMES = ["mix"]
GRAPH_GENERATORS = {}
for name, func in GRAPH_GENERATORS_PAIRS:
    GRAPH_GENERATOR_NAMES.append(name)
    GRAPH_GENERATORS[name] = func


# mixture of generators as in PNA (https://arxiv.org/pdf/2004.05718.pdf).
MIXTURE = {
    "er": 0.2,
    "ba": 0.2,
    "grid": 0.05,
    "caveman": 0.05,
    "tree": 0.15,
    "ladder": 0.05,
    "line": 0.05,
    "star": 0.05,
    "caterpillar": 0.1,
    "lobster": 0.1,
}


def get_random_graph_builder(method="mix"):
    if method == "mix":
        method = sample_from_mixture(MIXTURE)

    def graph_builder(n, degree=None, **kwargs):
        generator = GRAPH_GENERATORS[method]
        if method in ["er", "ba", "bloom"]:
            generator = partial(generator, degree=degree)
        ret = generator(n, **kwargs)
        if type(ret) is tuple:
            return ret[0]
        return ret

    return graph_builder


def generate_pseudotree(n_nodes: int):
    """[v1] Generate a random tree, then a random edge to form pseudotree."""
    g = nx.random_tree(n=n_nodes)
    edges = nx.dfs_edges(g, source=0)
    tree = nx.DiGraph(edges)

    n = nx.number_of_nodes(g)
    node_label = np.zeros(shape=(n))
    u, v = sample_random_edge(g)
    lca = nx.lowest_common_ancestor(tree, u, v)
    g.add_edge(u, v)

    def func(u, lca):
        l = []
        while u != lca:
            l.append(u)
            u = list(tree.predecessors(u))[0]
        return l

    idx = func(u, lca) + func(v, lca) + [lca]
    node_label[idx] = 1

    return g, node_label


from IPython import embed

if __name__ == "__main__":
    g, cycle = generate_graph_pseudotree(15, [0.3, 0.5], partition_method="iter")
    embed()
