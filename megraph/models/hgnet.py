#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : unet.py
# Author : Jiawei Xu
# Email  : 1138217508@qq.com
#
# Distributed under terms of the MIT license.

# Hierarchical Graph Net (https://arxiv.org/pdf/2107.07432.pdf)
# reproduce based on https://github.com/rampasek/HGNet

from functools import partial

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from megraph.dgl_utils import augment_graph_if_below_thresh
from megraph.layers import RGCNBlock
from megraph.pool import EdgePooling
from megraph.representation.features import MultiFeatures
from megraph.torch_utils import get_norm_layer
from megraph.utils import residual_when_same_shape
from torch_geometric.nn import MessagePassing, RGCNConv
from torch_geometric.utils import add_remaining_self_loops, degree, sort_edge_index

from . import register_function
from .model import MultiFeaturesModel


@register_function("hgnet")
class HGNet(MultiFeaturesModel):
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
        inter_connect="edge",
        # MultiFeaturesModel Args
        **kwargs,
    ):
        super(HGNet, self).__init__(
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
        self.num_layers = max_height
        self.inter_connect = inter_connect
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

        # up convs
        self.up_gns = nn.ModuleList()
        for i in range(1, self.max_height):
            conv = self.get_conv(i, down_current_dims, self.hidden_dims)
            self.up_gns.append(self.get_conv_block(conv))
            down_current_dims = conv.get_output_dims()

        # down convs
        self.down_gns = nn.ModuleList()
        for i in range(1, self.max_height):
            dim = self.hidden_dims[1]
            self.down_gns.append(self.get_rgcn_block(dim))

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
        self.weights = torch.nn.Parameter(torch.randn((self.max_height)))
        torch.nn.init.normal_(self.weights)

    def get_rgcn_block(self, dim, use_dropout=True):
        if self.inter_connect == "edge":
            conv = partial(RGCNConv, num_relations=2, aggr="add")
            dims = [0, dim, 0]
        elif self.inter_connect == "addnode":
            conv = GCNConvWParent
            dims = [0, dim, dim]
        norms = nn.ModuleList([get_norm_layer(self.norm_layer, hid) for hid in dims])
        drop = self.dropout if use_dropout else None
        return RGCNBlock(
            conv(dim, dim),
            norms=norms,
            act=self.activation,
            dropout=drop,
            inter_connect=self.inter_connect,
        )

    def forward(self, graph: dgl.DGLGraph):
        ori_graph = graph
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

        graphs_list = [graph]
        features_list = [features]
        edge_indices = [torch.stack(graph.edges(), dim=0)]
        clusters = []

        # Remove the self loop and fetch the edge feat.
        edge_feat = features.edges_features if self.pool_using_edge_feat else None
        if edge_feat is None:
            graph = dgl.remove_self_loop(graph)
        else:
            with graph.local_scope():
                graph.edata["x"] = features.edges_features
                graph = dgl.remove_self_loop(graph)
                edge_feat = graph.edata["x"]

        augmented = False
        # up flow
        for i, layer in enumerate(self.up_gns):
            pooling = self.poolings[0] if self.share_pool else self.poolings[i]

            # compose node features for pooling
            if pf_mode == "both" and pos_feat is not None:
                node_feat = torch.cat([features.nodes_features, pos_feat], dim=-1)
            elif use_pe_for_pooling:
                node_feat = pos_feat
            else:
                node_feat = features.nodes_features

            # pooling
            graph, _, cluster, edge_feat = pooling(graph, node_feat, edge_feat)

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

            # conv
            features = features.residual_when_same_shape(
                layer(graph, features),
                stem_beta=self.stem_beta,
                branch_beta=self.branch_beta,
            )
            features = self.apply_post_layer_oprs(features, ind=i + 1)

            graphs_list.append(graph)
            features_list.append(features)
            edge_indices.append(torch.stack(graph.edges(), dim=0))
            clusters.append(cluster)

        graphs_list = list(reversed(graphs_list))
        features_list = list(reversed(features_list))
        edge_indices = list(reversed(edge_indices))
        clusters = list(reversed(clusters))

        # up flow
        x = features_list[0][1]
        for i, layer in enumerate(self.down_gns):
            cluster = clusters[i]
            edge_index = edge_indices[i + 1]
            res = features_list[i + 1][1]
            if self.inter_connect == "edge":
                unpooled_num_nodes = res.shape[0]
                unpooled_num_edges = edge_index.size(1)
                x = torch.cat([res, x], dim=0)
                inter_edge_index = torch.stack(
                    [
                        torch.arange(unpooled_num_nodes),
                        cluster.cpu() + unpooled_num_nodes,
                    ]
                ).to(x.device)
                edge_index = torch.cat(
                    [edge_index, inter_edge_index, inter_edge_index[(1, 0),]], dim=1
                )
                edge_type = torch.cat(
                    [
                        torch.zeros(unpooled_num_edges, dtype=torch.long),
                        torch.ones(2 * unpooled_num_nodes, dtype=torch.long),
                    ],
                    dim=0,
                ).to(x.device)
                edge_index, edge_type = add_remaining_self_loops(
                    edge_index, edge_type, fill_value=0
                )
                edge_index, edge_type = sort_edge_index(
                    edge_index, edge_type, num_nodes=x.size(0)
                )
                features = layer(x, edge_index, edge_type, unpooled_num_nodes)
            elif self.inter_connect == "addnode":
                # up pool
                pooling = (
                    self.poolings[0]
                    if self.share_pool
                    else self.poolings[self.max_height - 1 - (i + 1)]
                )
                parent = pooling.unpool(x, graphs_list[i + 1], None, cluster)
                x = res
                edge_attr = features_list[i + 1][2]
                features = layer(x, parent, edge_index, edge_attr)
            x = features[1]
        logits = self.apply_last_layer(ori_graph, features)
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
            default=3,
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
        cls._add_argument(
            "inter_connect",
            "-ic",
            type=str,
            default="edge",
            choices=["edge", "addnode"],
            help="addnode for ogbg, edge for others",
        )


class GCNConvWParent(MessagePassing):
    """
    GCN convolution with a "parent" node
    """

    def __init__(self, in_dim, emb_dim, edge_encoder=None):
        super(GCNConvWParent, self).__init__(aggr="add")

        self.linear = nn.Linear(in_dim, emb_dim)
        self.root_emb = nn.Embedding(1, in_dim)
        self.parent_emb = nn.Embedding(1, in_dim)

        self.in_features = in_dim
        self.out_features = emb_dim
        self.edge_encoder = edge_encoder

    def forward(self, x, parent, edge_index, edge_attr=None):
        if self.edge_encoder is not None:
            edge_embedding = self.edge_encoder(edge_attr)
        else:
            edge_embedding = edge_attr
        if edge_embedding is None:
            edge_embedding = torch.zeros(
                (edge_index.size(1), 1), device=edge_index.device
            )

        row, col = edge_index

        deg = degree(row, x.size(0), dtype=x.dtype) + 2
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = (
            self.propagate(edge_index, x=x, edge_attr=edge_embedding, norm=norm)
            + F.relu(x + self.root_emb.weight) * 1.0 / deg.view(-1, 1)
            + F.relu(parent + self.parent_emb.weight) * 1.0 / deg.view(-1, 1)
        )
        out = self.linear(out)
        return out

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

    def __repr__(self) -> str:
        return "in_features={}, out_features={}".format(
            self.in_features, self.out_features
        )
