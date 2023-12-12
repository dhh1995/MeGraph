#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : synthetic_graph_tasks.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
#
# Distributed under terms of the MIT license.

"""
Register a set of synthetic graph theory tasks.
All dataset names start with task name,
    may followed by a suffix indicating variant targets.
The subname start with scale, i.e. one of [tiny, small, normal],
    followed by the name of graph generator.
The subname may have suffix indicate some important task-dependent args.

E.g. the dataset full name [sssp_greg_small_line_t5] means shortest path
graph regression task on small line graphs, with 5 tasks (channels) in parallel.

NOTE: To get comparable results across different machines, please use
--seed 2022 during the first time running a dataset (with same dname and subname).
The dataset will be cached and kept the same in the following runs.
"""

from argparse import Namespace
from functools import partial

import networkx as nx
import numpy as np
import numpy.random as random
from megraph.datasets.utils.graph_algorithms import (
    check_is_bipartite_graph,
    count_connected_components,
    get_diameter_length_and_nodes,
    get_disjoint_components_size,
)
from megraph.datasets.utils.graph_generators import (
    GRAPH_GENERATOR_NAMES,
    get_random_graph_builder,
)

from . import register_function
from .graph_theory import GraphTheoryDataset

# Available datasets
graph_theory_datasets = [
    "sssp_greg",
    "sssp_nreg",
    "reach_npred",
    "cc_greg",
    "diameter_npred",
    "diameter_greg",
    "ecc_nreg",
    "big_gpred",
    "cntc_greg",
]
dname = Namespace()
AVAILABLE_GT_DATASETS = []
for d in graph_theory_datasets:
    name = f"me_{d}"
    setattr(dname, d, name)
    AVAILABLE_GT_DATASETS.append(name)

# pre-set scales
GRAPH_SCALE_NGRAPHS = {
    "tiny": 200,
    "small": 300,
    "normal": 500,
}
GRAPH_SCALE_MIN_MAX = {
    "tiny": [5, 10],
    "small": [20, 50],
    "normal": [100, 200],
}
GRAPH_SCALE_NGRAPHS["custom"] = GRAPH_SCALE_NGRAPHS["normal"]
GRAPH_SCALE_MIN_MAX["custom"] = GRAPH_SCALE_MIN_MAX["normal"]
GRAPH_SCALES = GRAPH_SCALE_MIN_MAX.keys()

AVILABLE_NAMES = set()
for method in GRAPH_GENERATOR_NAMES:
    AVILABLE_NAMES.add(f"{method}")
for scale in GRAPH_SCALES:
    for method in GRAPH_GENERATOR_NAMES:
        AVILABLE_NAMES.add(f"{scale}_{method}")


def check_subname(name):
    for prefix in AVILABLE_NAMES:
        if name.startswith(prefix):
            return True
    raise ValueError(f"Invalid dataset subname {name}")


def get_args_from_extra(extra, keyward, default, single_value=True):
    for i in range(len(extra)):
        cur = extra[i]
        if cur.startswith(keyward):
            value = int(cur[len(keyward) :])
            if single_value:
                return value
            else:
                mi, ma = [value, value]
                try:
                    ma = int(extra[i + 1])
                except:
                    pass
                if mi > ma:
                    raise ValueError(f"The range [{mi}, {ma}] is not valid.")
                return [mi, ma]

    return default


def get_graph_info_from_extra(extra, default_n_graphs, default_n_min_max):
    n_graphs = get_args_from_extra(extra, "g", default_n_graphs)
    n_min_max = get_args_from_extra(extra, "n", default_n_min_max, single_value=False)
    degree_min_max = get_args_from_extra(extra, "d", None, single_value=False)
    return dict(
        n_graphs=n_graphs, num_nodes_min_max=n_min_max, degree_min_max=degree_min_max
    )


def get_info_from_subname(subname):
    check_subname(subname)
    splits = subname.split("_")
    if splits[0] in GRAPH_SCALES:
        scale, method = splits[:2]
        extra = splits[2:]
    else:
        scale, method = "custom", splits[0]
        extra = splits[1:]

    graph_builder = get_random_graph_builder(method)
    graph_info = get_graph_info_from_extra(
        extra, GRAPH_SCALE_NGRAPHS[scale], GRAPH_SCALE_MIN_MAX[scale]
    )
    return graph_builder, graph_info, extra


COMMON_META = {
    "inductive": True,
    "need_subname": True,
    "need_adapter": False,
}


# ** Begin definition of datasets
# -------------------- shortest path graph regression --------------------
def sssp_greg_feat_label(graph: nx.DiGraph, feat_size, num_tasks=1, num_tries=5):
    # TODO: what if graph is not connected?
    n = graph.number_of_nodes()
    feat = np.zeros((n, feat_size))
    distances = []
    num = 0
    feat[:, 0] = 1
    for _ in range(num_tries):
        src = random.permutation(n)
        dst = random.permutation(n)
        for s, d in zip(src, dst):
            try:
                distance = nx.shortest_path_length(graph, s, d)
            except:
                # NetworkXNoPath
                continue
            feat[s, num * 2 + 1] = 1
            feat[d, num * 2 + 2] = 1
            num += 1
            distances.append(distance)
            if num == num_tasks:
                return dict(feat=feat, glabel=distances)


@register_function(dname.sssp_greg, {"task": "gpred", "reg": True, **COMMON_META})
def sssp_greg(name, **kwargs):
    graph_builder, graph_info, extra = get_info_from_subname(name)
    num_tasks = get_args_from_extra(extra, "t", 1)
    feat_size = 1 + num_tasks * 2
    kwargs.update(graph_info)
    return GraphTheoryDataset(
        f"{dname.sssp_greg}_{name}",
        graph_builder,
        partial(sssp_greg_feat_label, num_tasks=num_tasks),
        num_tasks=num_tasks,
        feat_size=feat_size,
        **kwargs,
    )


# -------------------- shortest path node regression --------------------
def sssp_nreg_feat_label(graph: nx.DiGraph, feat_size, num_tasks=1):
    # TODO: what if graph is not connected?
    n = graph.number_of_nodes()
    feat = np.zeros((n, feat_size))
    feat[:, 0] = 1
    distances = []
    for i in range(num_tasks):
        s = random.randint(0, n)
        feat[s, i + 1] = 1

        distance = np.ones(n) * -10  # unreachable ones as -10
        dist = nx.shortest_path_length(graph, source=s)
        for k, v in dist.items():
            distance[k] = v
        distances.append(distance)
    distances = np.stack(distances, axis=1)
    return dict(feat=feat, nlabel=distances)


@register_function(dname.sssp_nreg, {"task": "npred", "reg": True, **COMMON_META})
def sssp_nreg(name, **kwargs):
    graph_builder, graph_info, extra = get_info_from_subname(name)
    num_tasks = get_args_from_extra(extra, "t", 1)
    feat_size = 1 + num_tasks
    kwargs.update(graph_info)
    return GraphTheoryDataset(
        f"{dname.sssp_nreg}_{name}",
        graph_builder,
        partial(sssp_nreg_feat_label, num_tasks=num_tasks),
        num_tasks=num_tasks,
        feat_size=feat_size,
        **kwargs,
    )


# -------------------- reachiability node regression --------------------
def reach_npred_feat_label(graph: nx.DiGraph, feat_size, num_tasks=1):
    n = graph.number_of_nodes()
    feat = np.zeros((n, feat_size))
    feat[:, 0] = 1
    reaches = []
    for i in range(num_tasks):
        s = random.randint(0, n)
        feat[s, i + 1] = 1

        reachable = np.zeros(n)
        reachable[list(nx.shortest_path_length(graph, source=s).keys())] = 1
        reaches.append(reachable)
    reaches = np.stack(reaches, axis=1)
    return dict(feat=feat, nlabel=reaches)


@register_function(dname.reach_npred, {"task": "npred", **COMMON_META})
def reach_npred(name, **kwargs):
    graph_builder, graph_info, extra = get_info_from_subname(name)
    num_tasks = get_args_from_extra(extra, "t", 1)
    feat_size = 1 + num_tasks
    kwargs.update(graph_info)
    return GraphTheoryDataset(
        f"{dname.reach_npred}_{name}",
        graph_builder,
        partial(reach_npred_feat_label, num_tasks=num_tasks),
        num_classes=2,
        num_tasks=num_tasks,
        feat_size=feat_size,
        **kwargs,
    )


# -------------------- connected colors graph regression --------------------
def cc_greg_feat_label(graph: nx.DiGraph, feat_size, num_colors=3):
    n = graph.number_of_nodes()
    c = num_colors
    feat = np.zeros((n, feat_size))
    colors = np.random.randint(c, size=n)

    # Add random paints to increase the count
    m = min(n, 10)
    for i in range(m):
        for j in range(c):
            start = np.random.randint(n)
            colors[start : start + n // m] = j

    feat[np.arange(n), colors] = 1
    nums = []
    for i in range(c):
        if np.sum(colors == i) == 0:
            nums.append(0)
        else:
            counts = get_disjoint_components_size(graph, colors == i)
            nums.append(max(counts))
    return dict(feat=feat, glabel=nums)


@register_function(dname.cc_greg, {"task": "gpred", "reg": True, **COMMON_META})
def cc_greg(name, **kwargs):
    graph_builder, graph_info, extra = get_info_from_subname(name)
    num_colors = 3
    if name.startswith("tiny"):
        num_colors = 2
    num_colors = get_args_from_extra(extra, "c", num_colors)
    kwargs.update(graph_info)
    return GraphTheoryDataset(
        f"{dname.cc_greg}_{name}",
        graph_builder,
        partial(cc_greg_feat_label, num_colors=num_colors),
        num_tasks=num_colors,
        feat_size=num_colors,
        **kwargs,
    )


# -------------------- diameter graph regression --------------------
def diameter_greg_feat_label(graph: nx.DiGraph, feat_size):
    diameter = get_diameter_length_and_nodes(graph, need_nodes_label=False)
    # TODO: what if graph is not connected?
    return dict(glabel=[diameter])


@register_function(dname.diameter_greg, {"task": "gpred", "reg": True, **COMMON_META})
def diameter_greg(name, **kwargs):
    graph_builder, graph_info, _ = get_info_from_subname(name)
    kwargs.update(graph_info)
    return GraphTheoryDataset(
        f"{dname.diameter_greg}_{name}",
        graph_builder,
        diameter_greg_feat_label,
        **kwargs,
    )


# -------------------- graph diameter node pred --------------------
def diameter_npred_feat_label(graph: nx.DiGraph, feat_size):
    diameter, nodes_label = get_diameter_length_and_nodes(graph, need_nodes_label=True)
    # TODO: what if graph is not connected? each connected components seperately
    return dict(nlabel=nodes_label)


@register_function(dname.diameter_npred, {"task": "npred", **COMMON_META})
def diameter_npred(name, **kwargs):
    graph_builder, graph_info, _ = get_info_from_subname(name)
    kwargs.update(graph_info)
    return GraphTheoryDataset(
        f"{dname.diameter_npred}_{name}",
        graph_builder,
        diameter_npred_feat_label,
        num_classes=2,
        **kwargs,
    )


# -------------------- eccentricity node reg --------------------
def ecc_nreg_feat_label(graph: nx.DiGraph, feat_size):
    distance = nx.all_pairs_shortest_path_length(graph)
    n = graph.number_of_nodes()
    eccentricity = np.zeros(n)
    for i, dist in distance:
        eccentricity[i] = np.max(list(dist.values()))
    return dict(nlabel=eccentricity)


@register_function(dname.ecc_nreg, {"task": "npred", "reg": True, **COMMON_META})
def ecc_nreg(name, **kwargs):
    graph_builder, graph_info, _ = get_info_from_subname(name)
    kwargs.update(graph_info)
    return GraphTheoryDataset(
        f"{dname.ecc_nreg}_{name}",
        graph_builder,
        ecc_nreg_feat_label,
        **kwargs,
    )


# -------------------- bipartite graph pred --------------------
def big_gpred_feat_label(graph: nx.DiGraph, feat_size, num_colors=1):
    m = graph.number_of_edges()
    efeat = np.zeros((m, num_colors))
    colors = np.random.randint(num_colors, size=m)
    efeat[np.arange(m), colors] = 1
    labels = []
    for c in range(num_colors):
        selected = efeat[:, c] == 1
        labels.append(check_is_bipartite_graph(graph, selected))
    return dict(efeat=efeat, glabel=labels)


@register_function(dname.big_gpred, {"task": "gpred", **COMMON_META})
def big_gpred(name, **kwargs):
    graph_builder, graph_info, extra = get_info_from_subname(name)
    num_colors = get_args_from_extra(extra, "c", 1)
    kwargs.update(graph_info)
    return GraphTheoryDataset(
        f"{dname.big_gpred}_{name}",
        graph_builder,
        partial(big_gpred_feat_label, num_colors=num_colors),
        num_classes=2,
        edge_feat_size=num_colors,
        **kwargs,
    )


# -------------------- Count Connected Components graph reg --------------------
def cntc_greg_feat_label(graph: nx.DiGraph, feat_size, num_colors=1):
    m = graph.number_of_edges()
    efeat = np.zeros((m, num_colors))
    colors = np.random.randint(num_colors, size=m)
    efeat[np.arange(m), colors] = 1
    labels = []
    for c in range(num_colors):
        selected = efeat[:, c] == 1
        labels.append(count_connected_components(graph, selected))
    return dict(efeat=efeat, glabel=labels)


@register_function(dname.cntc_greg, {"task": "gpred", "reg": True, **COMMON_META})
def cntc_greg(name, **kwargs):
    graph_builder, graph_info, extra = get_info_from_subname(name)
    num_colors = get_args_from_extra(extra, "c", 1)
    kwargs.update(graph_info)
    return GraphTheoryDataset(
        f"{dname.cntc_greg}_{name}",
        graph_builder,
        partial(cntc_greg_feat_label, num_colors=num_colors),
        edge_feat_size=num_colors,
        **kwargs,
    )
