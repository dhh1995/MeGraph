#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : manager.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
#
# Distributed under terms of the MIT license.

import os.path as osp
from functools import partial

import numpy as np
import torch
from dgl.data.adapter import AsLinkPredDataset, AsNodePredDataset
from megraph.dgl_utils import (add_reverse_edges, add_self_loop,
                               get_num_disjoint, positioal_encoding,
                               remove_self_loop)
from megraph.logger import logger
from sklearn.model_selection import StratifiedKFold

from .adapter import AsGraphPredDataset, InductiveNodePredDataset
from .cb_loss import CBLoss
from .evalutor import get_evaluator
from .utils import get_dataset_split, get_split_masks
from .wce_loss import WeightedCrossEntropyLoss

__all__ = ["graph_dataset_manager"]

ADAPTERS = dict(
    npred=AsNodePredDataset,
    lpred=AsLinkPredDataset,
    gpred=AsGraphPredDataset,
)
INDUCTIVEADAPTERS = dict(
    npred=InductiveNodePredDataset,
)

TASK_TARGET_DIM = {"gpred": 0, "npred": 1, "lpred": 2}


class DatasetManager(object):
    """The dataset manager for all graph-based datasets"""

    __dataset_manager_params__ = []
    __parser__ = None
    __params_set__ = False

    def __init__(self):
        self._datasets = {}

    def add_dataset(self, name, meta_data):
        """Add dataset to the manager

        Args:
            name (str): The name of the dataset
            meta_data (dict): The meta data of the dataset
        """
        default_task = meta_data["task"]
        ind_or_trans = "inductive" if meta_data["inductive"] else "transductive"
        print(f"Dataset registered: [{name}, {default_task}, {ind_or_trans}]")
        self._datasets[name] = meta_data

    def set_params_from_args(self, args):
        self.__params_set__ = True
        for k in self.__dataset_manager_params__:
            setattr(self, k, getattr(args, k))

    def load_and_process_dataset(self):
        """Load and process the dataset"""
        if not self.__params_set__:
            raise Exception("Params not set, please call set_params_from_args first.")

        name = self.dataset_name
        if name not in self._datasets.keys():
            raise ValueError(f"Unknown dataset: {name}")
        meta_data = self._datasets[name]

        raw_dir = None
        meta_data["raw_dir"] = raw_dir
        if self.task is None:
            self.task = meta_data["task"]

        func = meta_data["fn"]
        if meta_data.get("need_subname", False):
            func = partial(func, name=self.dataset_subname)
        dataset = func(raw_dir=raw_dir)
        if meta_data.get("reg", False) and not meta_data.get("loss_func", None):
            meta_data["loss_func"] = "MSE"
            meta_data["eval_metric"] = "L2"
        if "loss_func" not in meta_data.keys():
            meta_data["loss_func"] = "CE"
        meta_data["subg_sampling"] = self.subgraph_num > 1
        if self.subgraph_num > 1:
            meta_data["subg_num"] = self.subgraph_num
            meta_data["cache_path"] = osp.join(
                dataset.root, f"cluster_{self.subgraph_num}.pkl"
            )
        if hasattr(dataset, "meta_info"):
            # ogb
            meta_data["eval_metric"] = dataset.eval_metric
            if hasattr(dataset, "num_tasks"):
                meta_data["num_tasks"] = dataset.num_tasks
            if hasattr(dataset, "num_classes"):
                meta_data["num_classes"] = dataset.num_classes
            if dataset.task_type == "binary classification":
                meta_data["loss_func"] = "BCE"
            elif dataset.task_type == "regression":
                meta_data["loss_func"] = "MSE"
        if self.cb_loss_type is not None:
            meta_data["loss_func"] = "CB"
        self.loss_func = meta_data["loss_func"]
        if self.dataset_name == "ogbg" and self.dataset_subname.startswith("mol"):
            meta_data["embed_method"] = dict(node="mol")
        self.raw_dataset = dataset
        dataset = self.preprocess_dataset(dataset, meta_data)
        self.meta_data = meta_data
        self.dataset = dataset
        return dataset

    def preprocess_dataset(self, dataset, meta_data):
        train_ratio = self.train_ratio or 0.8
        val_ratio = self.val_ratio or 0.1
        test_ratio = 1.0 - train_ratio - val_ratio
        split_ratio = [train_ratio, val_ratio, test_ratio]
        if self.train_ratio is None and not self.enable_cross_validation:
            split_ratio = None  # Use dataset split
        raw_dir = meta_data["raw_dir"]
        need_node_feat = False
        if meta_data.get("need_adapter", True):
            if self.task == "gpred":
                dataset = ADAPTERS[self.task](dataset, split_ratio, raw_dir=raw_dir)
                # num_classes = dataset.num_tasks
                logger.info(
                    f"# Train samples {len(dataset.train_idx)}"
                    f", Val samples {len(dataset.val_idx)}"
                    f", Test samples {len(dataset.test_idx)}"
                )
                if dataset.node_feat_size is None:
                    need_node_feat = True
            else:
                try:
                    if meta_data.get("inductive", False):
                        dataset = INDUCTIVEADAPTERS[self.task](
                            dataset, split_ratio, raw_dir=raw_dir
                        )
                    else:
                        dataset = ADAPTERS[self.task](
                            dataset, split_ratio, raw_dir=raw_dir
                        )
                except Exception as e:
                    raise ValueError(
                        f"default dataset split not available for "
                        f"{self.dataset_name}, please provide train_ratio."
                    )
                # num_classes = dataset.num_classes
                dataset.node_feat_size = getattr(
                    dataset.dataset, "node_feat_size", None
                )
                if dataset.node_feat_size is None:
                    g = dataset[0]
                    dataset.node_feat_size = (
                        g.ndata["feat"].shape[-1] if "feat" in g.ndata else None
                    )  # FIXME when missing 'feat'
        num_graphs = len(dataset)
        num_nodes = []
        num_edges = []
        graphs = []
        labels = []
        for i in range(num_graphs):
            if self.task == "gpred":
                graph, label = dataset[i]
            else:
                graph = dataset[i]

            def prt(msg):
                if i % 1000 == 0:
                    logger.info(msg)

            if need_node_feat:
                graph.ndata["feat"] = torch.ones(graph.num_nodes(), 1).float()
            if "feat" in graph.ndata:
                graph.ndata["feat"] = graph.ndata["feat"].float()
            if "feat" in graph.edata:
                graph.edata["feat"] = graph.edata["feat"].float()
            if self.add_reverse_edges:
                prt(f"Num edges before adding reverse edge {graph.num_edges()}")
                graph = add_reverse_edges(graph)
                prt(f"Num edges after adding reverse edge {graph.num_edges()}")
            if self.to_simple_graph:
                prt(f"Num edges before to simple graph {graph.num_edges()}")
                graph = graph.to_simple()
                prt(f"Num edges after to simple graph {graph.num_edges()}")
            if self.self_loop:
                prt(f"Num edges before adding self-loop {graph.num_edges()}")
                graph = add_self_loop(remove_self_loop(graph))
                prt(f"Num edges after adding self-loop {graph.num_edges()}")
            if len(self.pe_types) > 0:
                # Note the pe is computed before multiple runs
                pes = [
                    positioal_encoding(graph, pe_type, self.pe_dim, self.pe_rep)
                    for pe_type in self.pe_types
                ]
                graph.ndata["pe"] = torch.cat(pes, dim=-1)
            if self.task == "gpred":
                if hasattr(dataset, "dataset"):
                    dataset.dataset.graphs[i] = graph
                graphs.append(graph)
                labels.append(label)
            num_nodes.append(graph.num_nodes())
            num_edges.append(graph.num_edges())

            prt(f"----Graph statistics------")
            prt(f"Graph {i}")
            prt(f"# Nodes {graph.num_nodes()}")
            prt(f"# Edges {graph.num_edges()}")

        if self.enable_cross_validation:
            if self.task != "gpred":
                raise ValueError(
                    "Cross validation is only available "
                    "for graph property prediction task."
                )
            kf = StratifiedKFold(
                n_splits=self.folds, shuffle=True, random_state=self.foldseed
            )
            self.cross_validation_splits = list(kf.split(graphs, labels))

        logger.info(f"# Graphs {num_graphs}")
        # logger.info(f"# Classes {num_classes}")
        logger.info(f"Avg Nodes {np.mean(num_nodes)}, Sum Nodes {np.sum(num_nodes)}")
        logger.info(f"Avg Edges {np.mean(num_edges)}, Sum Edges {np.sum(num_edges)}")
        logger.info("-" * 10 + "Done processing dataset" + "-" * 10)
        return dataset

    def get_dataset_and_meta_data(self):
        """Get dataset and meta data"""
        return self.dataset, self._datasets[self.dataset_name]

    def get_input_output_dim(self):
        """Get input and output dimensions for the dataset"""
        g_dim = 0
        n_dim = self.dataset.node_feat_size
        e_dim = (
            self.dataset.edge_feat_size or 0
            if hasattr(self.dataset, "edge_feat_size")
            else 0
        )
        input_dims = [g_dim, n_dim, e_dim]
        output_dims = [0, 0, 0]
        output_dims[TASK_TARGET_DIM[self.task]] = self.get_num_classes()
        pe_dim = len(self.pe_types) * self.pe_dim
        return input_dims, output_dims, pe_dim

    def get_loss_function(self):
        """Get loss function used for the dataset"""
        if self.loss_func == "CE":
            return torch.nn.CrossEntropyLoss()
        elif self.loss_func == "BCE":
            return torch.nn.BCEWithLogitsLoss()
        elif self.loss_func == "MSE":
            return torch.nn.MSELoss()
        elif self.loss_func == "L1":
            return torch.nn.L1Loss()
        elif self.loss_func == "WCE":
            return WeightedCrossEntropyLoss()
        elif self.loss_func == "CB":
            cbl_cfg = self.get_cb_loss_config()
            return CBLoss(**cbl_cfg)
        else:
            raise ValueError("Unknown Loss Type: {}".format(self.loss_func))

    def get_cb_loss_config(self):
        """Class balance loss config"""
        n_classes = self.dataset.num_classes
        if n_classes is None:  # TU dataset
            _, n_classes, _ = self.raw_dataset.statistics()
        no_of_classes = int(n_classes)
        if hasattr(self.raw_dataset, "labels"):
            labels = self.raw_dataset.labels.squeeze()
        elif hasattr(self.raw_dataset, "graph_labels"):
            labels = self.raw_dataset.graph_labels.squeeze()
        else:
            labels = self.dataset[0].ndata["label"]
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        samples_per_cls = [sum((labels == i)) for i in range(no_of_classes)]
        assert sum(samples_per_cls) == labels.shape[0]
        # class_ratio = np.array(samples_per_cls) / min(samples_per_cls)
        # print(f"class ratio: {class_ratio}")
        cbl_cfg = dict(
            samples_per_cls=samples_per_cls,
            no_of_classes=no_of_classes,
            beta=self.cb_loss_beta,
            gamma=self.cb_loss_gamma,
        )
        return cbl_cfg

    def get_evaluator(self):
        """Get evaluator for the dataset"""
        dataset_name = self.dataset_name
        if dataset_name.startswith("ogb"):
            dataset_name += f"-{self.dataset_subname}"
        return get_evaluator(
            dataset_name,
            self.task,
            self.cb_eval,
            num_classes=self.get_num_classes(),
            meta_data=self.meta_data,
        )

    def get_dataset_split(self, run_id=None):
        """Get dataset split for the dataset"""
        split_ratio = None
        if self.train_ratio is not None:
            split_ratio = {"train": self.train_ratio, "val": self.val_ratio}

        if self.enable_cross_validation:
            train, val = self.cross_validation_splits[run_id]
            train = torch.from_numpy(train)
            val = torch.from_numpy(val)
            return train, val, val

        if self.task == "gpred" or self.meta_data.get("inductive", False):
            if split_ratio is not None:
                return get_split_masks(len(self.dataset), split_ratio)
            return self.dataset.train_idx, self.dataset.val_idx, self.dataset.test_idx

        masks = get_dataset_split(self.dataset, split_ratio)
        splits = ["Train", "Val", "Test"]
        for split, mask in zip(splits, masks):
            logger.info(f"# {split} samples {mask.int().sum().item()}")
        return masks

    def get_num_classes(self):
        """Get number of classes for the dataset"""
        if "num_tasks" in self.meta_data:
            if self.meta_data["loss_func"] == "CE":
                # OGB
                nc = self.meta_data["num_classes"]
            else:
                nc = self.meta_data["num_tasks"]
            return int(nc)

        if self.meta_data.get("reg", False):
            # regression task
            n_classes = self.dataset.num_tasks
        else:
            n_classes = self.dataset.num_classes
        if n_classes is None:
            # TU dataset
            _, n_classes, _ = self.raw_dataset.statistics()
        if int(n_classes) == 1 and self.cb_loss_type is not None:
            n_classes = 2  # binary classification
        return int(n_classes)

    def _set_parser(self, parser):
        self.__parser__ = parser

    def _add_argument(self, name, *args, **kwargs):
        self.__dataset_manager_params__.append(name)
        name = name.replace("_", "-")
        self.__parser__.add_argument("--" + name, *args, **kwargs)

    def register_dataset_args(self, parser):
        self._set_parser(parser.add_argument_group("dataset"))
        self._add_argument(
            "task",
            "-task",
            type=str,
            default=None,
            choices=["npred", "epred", "gpred"],
            help="The task to run, can be Node/Edge/Graph Prediction",
        )
        self._add_argument(
            "dataset_name",
            "-dname",
            type=str,
            default="cora",
            choices=self._datasets.keys(),
            help="The input dataset",
        )
        self._add_argument(
            "dataset_subname",
            "-dsub",
            type=str,
            default=None,
            help="The name for the sub dataset, if applicable",
        )
        self._add_argument(
            "train_ratio",
            "-tr",
            type=float,
            default=None,
            help="The train ratio",
        )
        self._add_argument(
            "val_ratio",
            "-vr",
            type=float,
            default=0.1,
            help="The val ratio",
        )
        self._add_argument(
            "self_loop",
            "-sl",
            action="store_true",
            help="graph self-loop (default=False)",
        )
        self._add_argument(
            "add_reverse_edges",
            "-rev",
            action="store_true",
            help="Add reverse edges for graph (default=False)",
        )
        self._add_argument(
            "to_simple_graph",
            "-sg",
            action="store_true",
            help="transform to simple graph (default=False)",
        )
        self._add_argument(
            "pe_types",
            "-pes",
            type=str,
            nargs="+",
            default=[],
            choices=["laplacian", "random_walk", "bfs"],
            help="positional encoding types",
        )
        self._add_argument(
            "pe_dim", "-pdim", type=int, default=1, help="positional encoding dim"
        )
        self._add_argument(
            "pe_rep",
            "-prep",
            type=int,
            default=1,
            help="the number of repeat for positional encoding, only effective for bfs pe",
        )
        self._add_argument(
            "cb_loss_type",
            "-cbl",
            type=str,
            default=None,
            choices=["focal", "sigmoid", "softmax"],
            help="type of class balance loss function",
        )
        self._add_argument(
            "cb_loss_beta",
            "-cbb",
            type=float,
            default=0.9999,
            help="beta for class balance loss function",
        )
        self._add_argument(
            "cb_loss_gamma",
            "-cbg",
            type=float,
            default=2.0,
            help="gamma for class balance loss function",
        )
        self._add_argument(
            "cb_eval",
            "-cbe",
            action="store_true",
            help="use class balance evaluator (default=False)",
        )
        self._add_argument(
            "enable_cross_validation",
            "-ecv",
            action="store_true",
            help="enable cross validation",
        )
        self._add_argument(
            "folds",
            "-folds",
            type=int,
            default=10,
            help="number of folds for cross validation",
        )
        self._add_argument(
            "foldseed",
            "-fseed",
            type=int,
            default=None,
            help="The seed to control the split for cross validation",
        )
        self._add_argument(
            "subgraph_num",
            "-subgn",
            type=int,
            default=1,
            help="Subgraph num",
        )
