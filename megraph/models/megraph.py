#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : megraph.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
#
# Distributed under terms of the MIT license.

from collections import defaultdict

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data.utils import load_graphs, save_graphs
from megraph.dgl_utils import augment_graph_if_below_thresh
from megraph.layers import AttentionWeightLayer, MeeLayer
from megraph.pool import EdgePooling, LouvainPooling, RandomPooling
from megraph.representation.features import MultiFeatures
from megraph.utils import get_tuple_n, residual_when_same_shape

from . import register_function
from .model import MultiFeaturesModel


@register_function("megraph")
class MeGraph(MultiFeaturesModel):
    def __init__(
        self,
        input_dims,
        output_dims,
        pe_dim,
        task,
        build_conv,
        # MeGraph Args
        pooling_type="ep",
        pool_node_ratio=None,
        pool_degree_ratio=None,
        pool_topedge_ratio=None,
        cluster_size_limit=None,
        pool_feature_using="node",
        edgedrop=0.2,
        pool_noise_scale=None,
        cross_update_method="conv",
        pool_aggr_node="sum",
        pool_aggr_edge="sum",
        fully_thresh=None,
        stop_num=[],
        max_height=5,
        unet_like=False,
        start_heights=0,
        end_heights=None,
        readout_height=0,
        num_shared_convs=1,
        num_shared_pools=1,
        num_recurrent=None,
        stem_beta=1.0,
        branch_beta=0.5,
        keep_beta=1.0,
        cross_beta=0.5,
        dropout_after_residual=False,
        soft_readout=False,
        x_update_att=None,
        pool_with_cluster_score=False,
        unpool_with_cluster_score=False,
        # MultiFeaturesModel Args
        **kwargs,
    ):
        super(MeGraph, self).__init__(
            input_dims,
            output_dims,
            pe_dim,
            task,
            build_conv,
            **kwargs,
        )
        self.fully_thresh = fully_thresh
        self.stop_num = stop_num
        self.max_height = max_height
        self.stem_beta = stem_beta
        self.branch_beta = branch_beta
        self.soft_readout = soft_readout
        self.readout_height = readout_height
        self.x_update_att = x_update_att
        self.pool_using_edge_feat = pool_aggr_edge != "none"
        if pool_aggr_edge in ["none", "sum"]:
            pool_aggr_edge = "add"
        self.pool_feature_using = pool_feature_using
        self.pool_with_cluster_score = pool_with_cluster_score

        start_heights = get_tuple_n(start_heights, self.num_layers)
        end_heights = get_tuple_n(end_heights, self.num_layers)

        if unet_like:  # UNet-like structure
            # adjust heights interval
            self.num_layers = self.max_height * 2 - 1
            if self.num_heads is not None:
                self.num_heads = [self.num_heads[0]] * (self.num_layers + 1) + [1]
            start_heights = []
            end_heights = []
            for i in range(self.num_layers):
                h = i if i < self.max_height else self.num_layers - 1 - i
                start_heights.append(h)
                end_heights.append(h + 1)
            # cross_update_method = "pool"
            pool_with_cluster_score = True
            unpool_with_cluster_score = True

        # input layer
        current_dims = self.get_input_dims_after_encoder()
        # first layer
        conv = self.get_conv(0, current_dims, self.hidden_dims)
        self.first_conv = self.get_conv_block(conv)
        current_dims = conv.get_output_dims()
        pool_node_dim = current_dims[1]
        pool_edge_dim = current_dims[2] if self.pool_using_edge_feat else 0

        # pe for pool
        self.pool_conv = None
        if self.pool_feature_using in ["pe", "both"] and self.pos_enc_dim > 0:
            pos_enc_dims = [0, self.pos_enc_dim, 0]
            conv = self.get_conv(0, pos_enc_dims, self.hidden_dims)
            self.pool_conv = self.get_conv_block(conv)
            pool_node_dim = conv.get_output_dims()[1]
            if self.pool_feature_using == "both":
                pool_node_dim += current_dims[1]

        self.poolings = nn.ModuleList()
        self._pooling_type = pooling_type
        for i in range(min(num_shared_pools, max_height - 1)):
            if pooling_type == "ep":
                pooling = EdgePooling(
                    node_feat_dim=pool_node_dim,
                    edge_feat_dim=pool_edge_dim,
                    node_ratio=pool_node_ratio,
                    degree_ratio=pool_degree_ratio,
                    topedge_ratio=pool_topedge_ratio,
                    cluster_size_limit=cluster_size_limit,
                    dropout=edgedrop,
                    edge_aggr=pool_aggr_edge,
                    node_aggr=pool_aggr_node,
                    pool_with_cluster_score=pool_with_cluster_score,
                    unpool_with_cluster_score=unpool_with_cluster_score,
                    noise_scale=pool_noise_scale,
                )
            elif pooling_type == "lv":
                pooling = LouvainPooling(
                    edge_aggr=pool_aggr_edge,
                    node_aggr=pool_aggr_node,
                )
            elif pooling_type == "rand":
                pooling = RandomPooling(
                    node_ratio=pool_node_ratio,
                    edge_aggr=pool_aggr_edge,
                    node_aggr=pool_aggr_node,
                )
            else:
                raise ValueError(f"Unknown pooling type {pooling_type}")
            self.poolings.append(pooling)

        def get_convs(layer_ind, num, current_dims):
            convs = nn.ModuleList()
            for _ in range(num):
                conv = self.get_conv(layer_ind, current_dims, self.hidden_dims)
                convs.append(self.get_conv_block(conv, use_dropout=False))
            return convs

        def get_inter_cross_fusion_fcs(num, feat_dim, activation=None):
            if activation is None or activation == "none":
                return None
            fusion_fcs = nn.ModuleList()
            for _ in range(num):
                fusion_fcs.append(AttentionWeightLayer(feat_dim, activation))
            return fusion_fcs

        # hidden layers
        if num_recurrent is None or not (0 < num_recurrent <= self.num_layers):
            num_recurrent = self.num_layers
        self.num_recurrent = num_recurrent
        self.layers = nn.ModuleList()
        for i in range(1, num_recurrent + 1):
            num_inters = min(num_shared_convs, max_height - 1)
            intra_convs = get_convs(i, num_shared_convs, current_dims)
            inter_convs = get_convs(i, num_inters, current_dims)
            inter_fusion_fcs = get_inter_cross_fusion_fcs(
                num_shared_convs, current_dims[1], self.x_update_att
            )
            vertical_first = unet_like and (i > self.max_height)
            self.layers.append(
                MeeLayer(
                    intra_convs,
                    inter_convs=inter_convs,
                    pooling=self.poolings[0] if len(self.poolings) else None,
                    inter_fusion_fcs=inter_fusion_fcs,
                    cross_update_method=cross_update_method,
                    start_height=start_heights[i - 1],
                    end_height=None if end_heights is None else end_heights[i - 1],
                    vertical_first=vertical_first,
                    stem_beta=stem_beta,
                    branch_beta=branch_beta,
                    keep_beta=keep_beta,
                    cross_beta=cross_beta,
                    dropout=self.dropout,
                    dropout_after_residual=dropout_after_residual,
                )
            )
            current_dims = intra_convs[0].get_output_dims()
        self.prepare_last_layer(current_dims)
        self.weights = torch.nn.Parameter(torch.randn((self.num_layers)))
        torch.nn.init.normal_(self.weights)

    def get_pooling(self, i):
        maxh = self.max_height - 1
        if i < 0 or i >= maxh:
            raise ValueError("index must be in [0, max_height - 1)")
        return self.poolings[i * len(self.poolings) // maxh]

    def get_layer(self, i):
        if i < 0 or i >= self.num_layers:
            raise ValueError("index must be in [0, num_layers)")
        # NOTE: the input and output dimension should match when num_recurrent < num_layers
        return self.layers[i * self.num_recurrent // self.num_layers]

    def forward(self, graph: dgl.DGLGraph, save_graphs_filename=None):
        """The main forward function of MeGraph"""
        x = self.get_inputs(graph)

        # [optional, experimental] positional encoding for pooling
        pef = self.pos_enc
        pf_mode = self.pool_feature_using
        use_pe_for_pooling = pf_mode in ["pe", "both"] and pef is not None
        if use_pe_for_pooling:
            new_pef = self.pool_conv(graph, MultiFeatures([None, pef, None]))
            new_pef = new_pef.nodes_features
            pef = residual_when_same_shape(pef, new_pef, y_beta=self.branch_beta)
        pos_feat = pef
        x = x.residual_when_same_shape(
            self.first_conv(graph, x), branch_beta=self.branch_beta
        )
        x = self.apply_post_layer_oprs(x, ind=0)

        def should_stop(batch_num_nodes):
            """[Experimental] Check whether to stop pooling"""
            stop_num = self.stop_num
            metrics = ["mean", "max", "sum"]
            # avg/max/sum, stop when any of them satisfy the requirement
            for i, m in enumerate(metrics):
                nums = batch_num_nodes
                if m == "mean":
                    nums = nums.float()
                if len(stop_num) > i:
                    if getattr(torch, m)(nums) < stop_num[i]:
                        return True

            return False

        # down sample and build mega graph
        xs = [x]
        intra_graphs = [graph]
        inter_graphs = []
        clusters = []
        batch_num_nodes = graph.batch_num_nodes()

        # Remove the self loop and fetch the edge feat.
        edge_feat = x.edges_features if self.pool_using_edge_feat else None
        if edge_feat is None:
            graph = dgl.remove_self_loop(graph)
        else:
            with graph.local_scope():
                graph.edata["x"] = x.edges_features
                graph = dgl.remove_self_loop(graph)
                edge_feat = graph.edata["x"]

        augmented = False
        pooling_cache = {}
        while len(xs) < self.max_height and not should_stop(batch_num_nodes):
            # compose node features for pooling
            if pf_mode == "both" and pos_feat is not None:
                features = torch.cat([x.nodes_features, pos_feat], dim=-1)
            elif use_pe_for_pooling:
                features = pos_feat
            else:
                features = x.nodes_features

            # pooling
            pooling = self.get_pooling(len(inter_graphs))
            pooling_cache["height"] = len(inter_graphs)
            graph, inter_graph, cluster, edge_feat = pooling(
                graph, features, edge_feat, pooling_cache
            )
            node_feat = pooling.pool(x.nodes_features, graph, cluster)
            if use_pe_for_pooling:
                pos_feat = pooling.pool(pos_feat, graph, cluster, agg="mean")

            # Multiply the cluster scores onto features to get backward gradients
            if not self.pool_with_cluster_score:
                cluster_score = graph.ndata["cluster_score"].view(-1, 1)
                node_feat = node_feat * cluster_score
            if not augmented:
                graph, edge_feat, augmented = augment_graph_if_below_thresh(
                    graph, edge_feat, self.fully_thresh
                )

            x = MultiFeatures([None, node_feat, edge_feat])
            batch_num_nodes = graph.batch_num_nodes()

            xs.append(x)
            intra_graphs.append(graph)
            inter_graphs.append(inter_graph)
            clusters.append(cluster)

        # For debug and visulization
        if save_graphs_filename is not None:
            dump_graphs = defaultdict(list)
            for g in intra_graphs + inter_graphs:
                ug = dgl.unbatch(g)[:16]
                # save atmost 16 graphs for each batch
                for i, u in enumerate(ug):
                    dump_graphs[i].append(u)
            for k, v in dump_graphs.items():
                save_graphs(f"{save_graphs_filename}_{k:03d}.bin", v)

        height = len(xs)
        outs = []
        read_height = self.readout_height
        for i in range(self.num_layers):
            ind = i + 1
            xs = self.get_layer(i)(height, intra_graphs, inter_graphs, clusters, xs)
            # Mee Layer is about 3h x more expensive than a single conv
            xs = [
                self.apply_post_layer_oprs(x, ind=ind) for x in xs
            ]  # apply individually
            outs.append(xs[read_height])
        if self.soft_readout:
            weights = F.softmax(self.weights)
            x = sum([o * w for o, w in zip(outs, weights)])
        else:
            x = xs[read_height]
        x = self.apply_last_layer(intra_graphs[read_height], x)
        return x

    @classmethod
    def register_model_args(cls, parser, prefix=None):
        super().register_model_args(parser, prefix=prefix)
        cls._add_argument(
            "pooling_type",
            "-pt",
            type=str,
            default="ep",
            choices=["ep", "lv", "rand"],
            help="ep for EdgePool, lv for Louvain method, rand for random pooling.",
        )
        cls._add_argument(
            "pool_node_ratio",
            "-pnr",
            type=float,
            default=None,
            help="The ratio of num_nodes to be conserved for each pool",
        )
        cls._add_argument(
            "pool_degree_ratio",
            "-pdr",
            type=float,
            default=None,
            help="The maximum ratio of the edges (of degree) that being contracted",
        )
        cls._add_argument(
            "pool_topedge_ratio",
            "-per",
            type=float,
            default=None,
            help="The top edges to be considered in the pooling",
        )
        cls._add_argument(
            "cluster_size_limit",
            "-csl",
            type=int,
            default=None,
            help="The size limit of the cluster in the pooling",
        )
        cls._add_argument(
            "pool_feature_using",
            "-pfu",
            type=str,
            default="node",
            choices=["node", "pe", "both"],
            help="Use positional encoding, or node feat, or both as the feature for pooling",
        )
        cls._add_argument(
            "edgedrop", "-edrop", type=float, default=0.2, help="Edge score drop rate"
        )
        cls._add_argument(
            "pool_noise_scale",
            "-pns",
            type=float,
            default=None,
            help="the scale of noise that add on the scores for pooling",
        )
        cls._add_argument(
            "cross_update_method",
            "-xum",
            type=str,
            default="conv",
            choices=["conv", "pool", "combine"],
            help="The update method in cross update",
        )
        cls._add_argument(
            "pool_aggr_edge",
            "-pae",
            type=str,
            default="none",
            choices=["none", "sum", "mean", "max"],
            help="Pooling edge aggragator type",
        )
        cls._add_argument(
            "pool_aggr_node",
            "-pan",
            type=str,
            default="sum",
            choices=["sum", "mean", "max"],
            help="Pooling node aggragator type",
        )
        cls._add_argument(
            "stem_beta",
            "-sb",
            type=float,
            default=1.0,
            help="Beta for Stem in residual",
        )
        cls._add_argument(
            "branch_beta",
            "-bb",
            type=float,
            default=0.5,
            help="Beta for Branch in residual",
        )
        cls._add_argument(
            "keep_beta",
            "-kb",
            type=float,
            default=1.0,
            help="Beta for keep item in cross update of Mee Layer",
        )
        cls._add_argument(
            "cross_beta",
            "-xb",
            type=float,
            default=0.5,
            help="Beta for cross item in cross update of Mee Layer",
        )
        cls._add_argument(
            "fully_thresh",
            "-ft",
            type=float,
            default=None,
            help=(
                "Augment the pooled graph to fully connected when the "
                "average squared size below the threshold."
            ),
        )
        cls._add_argument(
            "stop_num",
            "-sn",
            type=int,
            nargs="+",
            default=[],
            help=(
                "Stop pooling when the num_nodes of graph (avg/max/sum across the batch)"
                "smaller than this number. Use 0 to represent not check."
            ),
        )
        cls._add_argument(
            "max_height",
            "-mh",
            type=int,
            default=5,
            help="Max height of the mega graph",
        )
        cls._add_argument(
            "readout_height",
            "-rh",
            type=int,
            default=0,
            help="The height of the mega graph that feed into last layer",
        )
        cls._add_argument(
            "unet_like",
            "-un",
            action="store_true",
            help="Implement unet_like structure by controlling start and end heights",
        )
        cls._add_argument(
            "start_heights",
            "-sth",
            type=int,
            nargs="+",
            default=0,
            help="Start height for each layer of the mega graph",
        )
        cls._add_argument(
            "end_heights",
            "-eth",
            type=int,
            nargs="+",
            default=None,
            help="End height for each layer of the mega graph",
        )
        cls._add_argument(
            "num_shared_convs",
            "-nsc",
            type=int,
            default=1,
            help="The number of convs to be shared among different heights",
        )
        cls._add_argument(
            "num_shared_pools",
            "-nsp",
            type=int,
            default=1,
            help="The number of poolings to be shared among different heights",
        )
        cls._add_argument(
            "num_recurrent",
            "-nr",
            type=int,
            default=None,
            help="The number of mee layers to be shared among different layers",
        )
        cls._add_argument(
            "dropout_after_residual",
            "-dar",
            action="store_true",
            help="place dropout after residual",
        )
        cls._add_argument(
            "soft_readout", "-sr", action="store_true", help="readout use softmax"
        )
        cls._add_argument(
            "x_update_att",
            "-xua",
            type=str,
            default=None,
            choices=["none", "softmax", "tanh", "sigmoid", "relu", "identity"],
            help="use cross update attention or not",
        )
        # Pool/Unpool with edge score is to compare whether to mul edge score when doing cross pool
        cls._add_argument(
            "pool_with_cluster_score",
            "-pcs",
            action="store_true",
            help=(
                "pool with cluster score during cross update, "
                "effect when cross update method is pool"
            ),
        )
        cls._add_argument(
            "unpool_with_cluster_score",
            "-ucs",
            action="store_true",
            help=(
                "unpool with cluster score during cross update, "
                "effect when cross update method is pool"
            ),
        )
