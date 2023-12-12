#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : unet.py
# Author : Jiawei Xu
# Email  : 1138217508@qq.com
#
# Distributed under terms of the MIT license.

# reproduce based on https://github.com/HongyangGao/Graph-U-Nets

import dgl
import torch
import torch.nn as nn
from megraph.dgl_utils import augment_graph_if_below_thresh
from megraph.pool import EdgePooling
from megraph.representation.features import MultiFeatures
from megraph.utils import residual_when_same_shape

from . import register_function
from .model import MultiFeaturesModel


@register_function("unet")
class UNetModel(MultiFeaturesModel):
    def __init__(
        self,
        input_dims,
        output_dims,
        pe_dim,
        task,
        build_conv,
        stem_beta=1.0,
        branch_beta=0.5,
        soft_readout=False,
        max_height=5,
        # egde pool args
        pool_node_ratio=None,
        pool_degree_ratio=None,
        pool_topedge_ratio=None,
        cluster_size_limit=None,
        pool_feature_using="node",
        edgedrop=0.2,
        pool_noise_scale=None,
        pool_aggr_node="sum",
        pool_aggr_edge="sum",
        fully_thresh=None,
        pool_with_cluster_score=False,
        unpool_with_cluster_score=False,
        share_pool=False,
        # MultiFeaturesModel Args
        **kwargs,
    ):
        super(UNetModel, self).__init__(
            input_dims,
            output_dims,
            pe_dim,
            task,
            build_conv,
            **kwargs,
        )
        self.stem_beta = stem_beta
        self.branch_beta = branch_beta
        self.soft_readout = soft_readout
        self.max_height = max_height
        self.num_layers = max_height * 2
        current_dims = self.get_input_dims_after_encoder()

        if self.num_heads is not None:
            self.num_heads = [self.num_heads[0]] * self.num_layers + [1]

        if self.use_scales:
            self.scales = nn.Parameter(torch.empty(self.num_layers + 1))
            nn.init.ones_(self.scales)

        # first conv
        conv = self.get_conv(0, current_dims, self.hidden_dims)
        self.first_conv = self.get_conv_block(conv)
        down_current_dims = conv.get_output_dims()
        up_current_dims = conv.get_output_dims()
        pool_current_dims = conv.get_output_dims()

        # down convs
        self.down_gns = nn.ModuleList()
        for i in range(1, self.max_height):
            conv = self.get_conv(i, down_current_dims, self.hidden_dims)
            self.down_gns.append(self.get_conv_block(conv))
            down_current_dims = conv.get_output_dims()

        # bottom conv
        conv = self.get_conv(self.max_height, down_current_dims, self.hidden_dims)
        self.last_conv = self.get_conv_block(conv)

        # up convs
        self.up_gns = nn.ModuleList()
        for i in range(1, self.max_height):
            conv = self.get_conv(i + self.max_height, up_current_dims, self.hidden_dims)
            self.up_gns.append(self.get_conv_block(conv))
            up_current_dims = conv.get_output_dims()

        self.fully_thresh = fully_thresh
        self.pool_using_edge_feat = pool_aggr_edge != "none"
        if pool_aggr_edge in ["none", "sum"]:
            pool_aggr_edge = "add"
        self.pool_feature_using = pool_feature_using
        pool_node_dim = pool_current_dims[1]
        pool_edge_dim = pool_current_dims[2] if self.pool_using_edge_feat else 0
        # pe for pool
        self.pool_conv = None
        if self.pool_feature_using in ["pe", "both"] and self.pos_enc_dim > 0:
            pos_enc_dims = [0, self.pos_enc_dim, 0]
            conv = self.get_conv(0, pos_enc_dims, self.hidden_dims)
            self.pool_conv = self.get_conv_block(conv)
            pool_node_dim = conv.get_output_dims()[1]
            if self.pool_feature_using == "both":
                pool_node_dim += pool_current_dims[1]
        # poolings
        self.share_pool = share_pool
        num_poolings = 1 if share_pool else self.max_height - 1
        self.poolings = nn.ModuleList()
        for i in range(num_poolings):
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
            self.poolings.append(pooling)

        # readout layer
        current_dims = up_current_dims
        self.prepare_last_layer(current_dims)
        self.weights = torch.nn.Parameter(torch.randn((self.max_height + 1)))
        torch.nn.init.normal_(self.weights)

    def forward(self, graph: dgl.DGLGraph):
        features = self.get_inputs(graph)

        # pos enc for pool
        pef = self.pos_enc
        pf_mode = self.pool_feature_using
        use_pe_for_pooling = pf_mode in ["pe", "both"] and pef is not None
        if use_pe_for_pooling:
            new_pef = self.pool_conv(graph, MultiFeatures([None, pef, None]))
            new_pef = new_pef.nodes_features
            pef = residual_when_same_shape(
                pef, new_pef, x_beta=self.stem_beta, y_beta=self.branch_beta
            )
        pos_feat = pef

        # first conv
        features = features.residual_when_same_shape(
            self.first_conv(graph, features),
            stem_beta=self.stem_beta,
            branch_beta=self.branch_beta,
        )
        features = self.apply_post_layer_oprs(features, ind=0)

        # Remove the self loop and fetch the edge feat.
        edge_feat = features.edges_features if self.pool_using_edge_feat else None
        if edge_feat is None:
            graph = dgl.remove_self_loop(graph)
        else:
            with graph.local_scope():
                graph.edata["x"] = features.edges_features
                graph = dgl.remove_self_loop(graph)
                edge_feat = graph.edata["x"]

        features_list = [features]
        graphs_list = [graph]
        clusters = []
        outs = []

        augmented = False
        # donw flow
        for i, layer in enumerate(self.down_gns):
            pooling = self.poolings[0] if self.share_pool else self.poolings[i]
            # conv
            features = features.residual_when_same_shape(
                layer(graph, features),
                stem_beta=self.stem_beta,
                branch_beta=self.branch_beta,
            )
            features = self.apply_post_layer_oprs(features, ind=i + 1)

            # compose node features for pooling
            if pf_mode == "both" and pos_feat is not None:
                node_feat = torch.cat([features.nodes_features, pos_feat], dim=-1)
            elif use_pe_for_pooling:
                node_feat = pos_feat
            else:
                node_feat = features.nodes_features

            # down pool
            graph, _, cluster, edge_feat = pooling(graph, node_feat, edge_feat)

            features_list.append(features)
            graphs_list.append(graph)
            clusters.append(cluster)

            # feature after pool
            node_feat = pooling.pool(features.nodes_features, graph, cluster)
            if use_pe_for_pooling:
                pos_feat = pooling.pool(pos_feat, graph, cluster, agg="mean")
            # Multiply the cluster scores onto features to get backward gradients
            if not pooling.pool_with_cluster_score:
                cluster_score = graph.ndata["cluster_score"].view(-1, 1)
                node_feat = node_feat * cluster_score
            if not augmented:
                graph, edge_feat, augmented = augment_graph_if_below_thresh(
                    graph, edge_feat, self.fully_thresh
                )
            features = MultiFeatures([None, node_feat, edge_feat])

        features_list = list(reversed(features_list))
        graphs_list = list(reversed(graphs_list))
        clusters = list(reversed(clusters))

        # bottom conv
        features = features.residual_when_same_shape(
            self.last_conv(graph, features),
            stem_beta=self.stem_beta,
            branch_beta=self.branch_beta,
        )
        features = self.apply_post_layer_oprs(features, ind=self.max_height + 1)

        # up flow
        for i, layer in enumerate(self.up_gns):
            pooling = self.poolings[0] if self.share_pool else self.poolings[i]
            # up pool
            new_node_feat = pooling.unpool(
                features.nodes_features,
                graphs_list[i],
                None,
                clusters[i],
            )
            features = MultiFeatures(
                [
                    None,
                    new_node_feat,
                    features_list[i].edges_features,
                ]
            )

            # conv
            features = features.residual_when_same_shape(
                layer(graphs_list[i + 1], features),
                stem_beta=self.stem_beta,
                branch_beta=self.branch_beta,
            )
            features = self.apply_post_layer_oprs(features, ind=self.max_height + i + 2)

            # add down conv feature
            node_feat = features.nodes_features + features_list[i].nodes_features
            edge_feat = features.edges_features
            features = MultiFeatures([None, node_feat, edge_feat])

            outs.append(features)

        # add fist conv feature
        node_feat = features.nodes_features + features_list[-1].nodes_features
        edge_feat = features.edges_features

        features = MultiFeatures([None, node_feat, edge_feat])
        outs.append(features)
        logits = self.apply_last_layer(graphs_list[-1], features)
        return logits

    @classmethod
    def register_model_args(cls, parser, prefix=None):
        super().register_model_args(parser, prefix=prefix)
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
            "soft_readout", "-sr", action="store_true", help="readout use softmax"
        )
        cls._add_argument(
            "max_height",
            "-mh",
            type=int,
            default=5,
            help="Max height of the mega graph",
        )
        # Edge pool
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
            "fully_thresh",
            "-ft",
            type=float,
            default=None,
            help=(
                "Augment the pooled graph to fully connected when the "
                "average squared size below the threshold."
            ),
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
        cls._add_argument(
            "share_pool",
            "-usp",
            action="store_true",
            help="unet share all poolings",
        )
