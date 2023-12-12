#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : ogb.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
#
# Distributed under terms of the MIT license.

import os.path as osp
from functools import partial

from ogb.graphproppred import DglGraphPropPredDataset
from ogb.linkproppred import DglLinkPropPredDataset
from ogb.nodeproppred import DglNodePropPredDataset

from . import graph_dataset_manager as manager

DEFAULT_PATH = osp.expanduser("~/.dgl")
OGB_DATASETS = {
    "npred": [
        "ogbn",
        # "ogbn-products",
        # "ogbn-proteins",
        # "ogbn-arxiv",
    ],
    "lpred": [
        "ogbl",
        # "ogbl-ppa",
        # "ogbl-collab",
        # "ogbl-ddi",
        # "ogbl-citation2",
        # "ogbl-wikikg2",
        # "ogbl-biokg",
        # "ogbl-vessel",
    ],
    "gpred": [
        "ogbg",
        # "ogbg-molhiv",
        # "ogbg-molpcba",
        # "ogbg-molbace",
        # "ogbg-molbbbp",
        # "ogbg-molclintox",
        # "ogbg-molmuv",
        # "ogbg-molsider",
        # "ogbg-moltox21",
        # "ogbg-moltoxcast",
        # "ogbg-molesol",
        # "ogbg-molfreesolv",
        # "ogbg-mollipo",
    ],
}


def get_npred_dataset(name, raw_dir):
    raw_dir = raw_dir or DEFAULT_PATH
    raw_dir = osp.join(raw_dir, "ogb")
    name = f"ogbn-{name}"
    dataset = DglNodePropPredDataset(name=name, root=raw_dir)
    return dataset


def get_lpred_dataset(name, raw_dir):
    raw_dir = raw_dir or DEFAULT_PATH
    raw_dir = osp.join(raw_dir, "ogb")
    name = f"ogbl-{name}"
    dataset = DglLinkPropPredDataset(name=name, root=raw_dir)
    return dataset


def get_gpred_dataset(name, raw_dir):
    raw_dir = raw_dir or DEFAULT_PATH
    raw_dir = osp.join(raw_dir, "ogb")
    name = f"ogbg-{name}"
    dataset = DglGraphPropPredDataset(name=name, root=raw_dir)
    return dataset


for name in OGB_DATASETS["npred"]:
    func = partial(get_npred_dataset, name=name)
    manager.add_dataset(
        name, dict(fn=func, task="npred", inductive=False, need_subname=True)
    )
for name in OGB_DATASETS["lpred"]:
    func = partial(get_lpred_dataset, name=name)
    manager.add_dataset(
        name, dict(fn=func, task="lpred", inductive=False, need_subname=True)
    )
for name in OGB_DATASETS["gpred"]:
    func = partial(get_gpred_dataset, name=name)
    manager.add_dataset(
        name, dict(fn=func, task="gpred", inductive=True, need_subname=True)
    )
