#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : layers.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
#
# Distributed under terms of the MIT license.

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
from dgl import DGLGraph
from dgl.nn.functional import edge_softmax
from dgl.nn.pytorch import (AGNNConv, APPNPConv, ChebConv, GATv2Conv, GINConv,
                            GraphConv, SAGEConv, SGConv, TAGConv)
from dgl.nn.pytorch.conv.pnaconv import AGGREGATORS, SCALERS, PNAConvTower
from dgl.utils import expand_as_pair
from megraph.pool.globalpool import get_global_edge_pooling, get_global_pooling
from megraph.representation import MultiFeatures
from megraph.torch_utils import apply_trans, sum_not_none_elements
from megraph.utils import apply_fn_on_list

from . import register_function
from .base import BaseGraphLayer


@register_function("gcn")
class GCNLayer(BaseGraphLayer):
    def __init__(self, input_dims, output_dims, **kwargs):
        super(GCNLayer, self).__init__(output_dims)
        in_feats = input_dims[1]
        out_feats = output_dims[1]
        self.conv = GraphConv(in_feats, out_feats, **kwargs)

    def update_nodes(self, graph: DGLGraph, features: MultiFeatures):
        return features.replace_nodes_features(
            self.conv(graph, features.nodes_features)
        )


@register_function("gat")
class GATLayer(BaseGraphLayer):
    def __init__(self, input_dims, output_dims, num_heads, **kwargs):
        super(GATLayer, self).__init__(output_dims)
        in_feats = input_dims[1]
        out_feats = output_dims[1]  # this should be the total dims (multiple heads)
        if out_feats % num_heads != 0:
            raise ValueError("output_dim of GAT Layer must be divisible by num_heads")
        out_feats = out_feats // num_heads
        self.conv = GATv2Conv(in_feats, out_feats, num_heads, **kwargs)

    def update_nodes(self, graph: DGLGraph, features: MultiFeatures):
        return features.replace_nodes_features(
            self.conv(graph, features.nodes_features).flatten(1)
        )

    @classmethod
    def register_layer_args(cls, parser, prefix=None):
        cls._set_parser_and_prefix(parser, prefix)
        cls._add_argument(
            "feat_drop", "-featdr", type=float, default=0.0, help="Feat Drop rate"
        )
        cls._add_argument(
            "attn_drop", "-attdr", type=float, default=0.0, help="Attn Drop rate"
        )
        cls._add_argument(
            "negative_slope",
            "-negsl",
            type=float,
            default=0.2,
            help="Negative slope of elu",
        )


@register_function("sage")
class SAGELayer(BaseGraphLayer):
    """GraphSage: Note that you need to change the sampler."""

    def __init__(self, input_dims, output_dims, aggregator_type, feat_drop, **kwargs):
        super(SAGELayer, self).__init__(output_dims)
        in_feats = input_dims[1]
        out_feats = output_dims[1]
        self.conv = SAGEConv(
            in_feats, out_feats, aggregator_type, feat_drop=feat_drop, **kwargs
        )

    def update_nodes(self, graph: DGLGraph, features: MultiFeatures):
        return features.replace_nodes_features(
            self.conv(graph, features.nodes_features)
        )

    @classmethod
    def register_layer_args(cls, parser, prefix=None):
        cls._set_parser_and_prefix(parser, prefix)
        cls._add_argument("aggregator_type", type=str, default="gcn", help="Aggregator")
        cls._add_argument("feat_drop", type=float, default=0.5, help="Drop rate")


@register_function("pna")
class PNALayer(BaseGraphLayer):
    """PNAConv, but failed"""

    def __init__(
        self, input_dims, output_dims, pna_aggregators, pna_scalers, pna_delta, **kwargs
    ):
        super(PNALayer, self).__init__(output_dims)
        in_feats = input_dims[1]
        out_feats = output_dims[1]
        self.conv = PNAConvTower(
            in_feats, out_feats, pna_aggregators, pna_scalers, pna_delta, **kwargs
        )

    def update_nodes(self, graph: DGLGraph, features: MultiFeatures):
        return features.replace_nodes_features(
            self.conv(graph, features.nodes_features)
        )

    @classmethod
    def register_layer_args(cls, parser, prefix=None):
        cls._set_parser_and_prefix(parser, prefix)
        cls._add_argument(
            "pna_aggregators",
            "-pnaag",
            type=str,
            choices=list(AGGREGATORS.keys()),
            default=["max", "mean", "sum"],
            nargs="+",
            help="Aggregators",
        )
        cls._add_argument(
            "pna_scalers",
            "-pnasc",
            type=str,
            choices=list(SCALERS.keys()),
            default=list(SCALERS.keys()),
            nargs="+",
            help="Scalers",
        )
        cls._add_argument("pna_delta", "-pnadt", type=float, default=2.5, help="Delta")


@register_function("gfn")
class GFNLayer(BaseGraphLayer):
    """Full GN layer as described in https://arxiv.org/abs/1806.01261 ."""

    def __init__(
        self,
        input_dims,
        output_dims,
        enpools,
        ngpools,
        egpools,
        disables=[],
        copies=[],
        num_heads=1,
        attn_drop=0.0,
        negative_slope=0.2,
        layer_bias=False,
    ):
        super(GFNLayer, self).__init__(output_dims)
        self.disables = disables
        self.copies = copies
        self.layer_bias = layer_bias
        g_idim, n_idim, e_idim = input_dims[:3]
        g_odim, n_odim, e_odim = output_dims[:3]
        # Gate
        if "gatesum" in enpools:
            self.ns_scorer = nn.Linear(n_idim, 1)
            self.nd_scorer = nn.Linear(n_idim, 1)

        # edges updates
        self.g2e = self.get_transform(g_idim, e_odim, "g2e")
        self.ns2e = self.get_transform(n_idim, e_odim, "ns2e")
        self.nd2e = self.get_transform(n_idim, e_odim, "nd2e")
        self.e2e = self.get_transform(e_idim, e_odim, "e2e")
        if g_idim + n_idim + e_idim <= 0:
            e_odim = 0
        self.output_dims[2] = e_odim
        e_idim = e_odim

        # nodes updates
        self.g2n = self.get_transform(g_idim, n_odim, "g2n")
        self.n2n = self.get_transform(n_idim, n_odim, "n2n")
        self.e2n = self.get_transform(e_idim * len(enpools), n_odim, "e2n")
        self.enpools = enpools
        if g_idim + n_idim + e_idim <= 0:
            n_odim = 0
        self.output_dims[1] = n_odim
        n_idim = n_odim

        # global updates
        self.g2g = self.get_transform(g_idim, g_odim, "g2g")
        self.n2g = self.get_transform(n_idim, g_odim, "n2g")
        self.e2g = self.get_transform(e_idim, g_odim, "e2g")
        self.ngpools = [get_global_pooling(ngpool) for ngpool in ngpools]
        self.egpools = [get_global_edge_pooling(egpool) for egpool in egpools]
        if g_idim + n_idim + e_idim <= 0:
            g_odim = 0
        self.output_dims[0] = g_odim

        self.e_odim = e_odim
        # Attention
        self.attn = None
        if "att" in enpools and e_odim > 0:
            if e_odim % num_heads != 0:
                raise ValueError("e_odim must be divisible by num_heads")
            self.num_heads = num_heads
            self.leaky_relu = nn.LeakyReLU(negative_slope)
            self.attn = nn.Parameter(
                torch.FloatTensor(size=(1, num_heads, e_odim // num_heads))
            )
            self.attn_drop = nn.Dropout(attn_drop)

        self.reset_attention_weight()

    def get_transform(self, input_dim, output_dim, name=None):
        if input_dim <= 0 or output_dim <= 0 or (name in self.disables):
            return None
        if input_dim == output_dim and (name in self.copies):
            return nn.Identity()
        linear = nn.Linear(input_dim, output_dim, bias=self.layer_bias)
        return linear

    def reset_attention_weight(self):
        gain = nn.init.calculate_gain("relu")
        if self.attn is not None:
            nn.init.xavier_normal_(self.attn, gain=gain)

    def reset_parameters(self):
        self.reset_attention_weight()
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    def mul_edge_weights(self, graph: DGLGraph, ef, nf=None, method="att"):
        # compute the edge weight with different methods
        if method == "att":
            ef = ef.view(-1, self.num_heads, self.e_odim // self.num_heads)
            x = self.leaky_relu(ef)  # (E, H, D)
            x = (x * self.attn).sum(dim=-1)  # (E, H)
            x = edge_softmax(graph, x).unsqueeze(dim=-1)  # (E, H, 1)
            ef = ef * self.attn_drop(x)
            ef = ef.view(-1, self.e_odim)
        elif method == "gatesum":
            if nf is not None:
                feat_src, feat_dst = expand_as_pair(nf, graph)
                graph.ndata["s_src"] = self.ns_scorer(feat_src)
                graph.ndata["s_dst"] = self.nd_scorer(feat_dst)
                graph.apply_edges(fn.u_add_v("s_src", "s_dst", "ew"))
                ef = ef * torch.sigmoid(graph.edata["ew"])
        return ef

    def update_edges(self, graph: DGLGraph, features: MultiFeatures):
        gf, nf, ef = features.get_global_nodes_edges_features()
        from_g = apply_trans(self.g2e, gf)
        if from_g is not None:
            from_g = dgl.broadcast_edges(graph, from_g)
        from_n = None
        from_e = apply_trans(self.e2e, ef)
        if nf is not None and self.ns2e is not None:
            with graph.local_scope():
                feat_src, feat_dst = expand_as_pair(nf, graph)
                graph.ndata["src"] = apply_trans(self.ns2e, feat_src)
                if self.nd2e is None:
                    graph.apply_edges(fn.copy_u("src", "ex"))
                else:
                    graph.ndata["dst"] = apply_trans(self.nd2e, feat_dst)
                    graph.apply_edges(fn.u_add_v("src", "dst", "ex"))
                from_n = graph.edata["ex"]
        return features.replace_edges_features(
            sum_not_none_elements([from_g, from_n, from_e])
        )

    def update_nodes(self, graph: DGLGraph, features: MultiFeatures):
        gf, nf, ef = features.get_global_nodes_edges_features()
        from_g = apply_trans(self.g2n, gf)
        if from_g is not None:
            from_g = dgl.broadcast_nodes(graph, from_g)
        from_n = apply_trans(self.n2n, nf)
        from_e = None

        if ef is not None and self.e2n is not None:
            with graph.local_scope():
                pooled_f = []
                for pool in self.enpools:
                    if pool in ["att", "gatesum"]:
                        graph.edata["ex"] = self.mul_edge_weights(
                            graph, ef, nf, method=pool
                        )
                        agg_func = fn.sum
                    else:
                        graph.edata["ex"] = ef
                        agg_func = getattr(fn, pool)
                    graph.update_all(fn.copy_e("ex", "m"), agg_func("m", f"{pool}_x"))
                    pooled_f.append(graph.ndata[f"{pool}_x"])
                from_e = apply_trans(self.e2n, torch.cat(pooled_f, dim=-1))
        return features.replace_nodes_features(
            sum_not_none_elements([from_g, from_n, from_e])
        )

    def update_global(self, graph: DGLGraph, features: MultiFeatures):
        gf, nf, ef = features.get_global_nodes_edges_features()
        from_g = apply_trans(self.g2g, gf)

        def apply_poolings_and_trans(x, poolings, trans):
            if len(poolings) == 0:
                return None
            if x is not None and trans is not None:
                x = torch.cat([pool(graph, x) for pool in poolings], dim=-1)
            return apply_trans(trans, x)

        from_n = apply_poolings_and_trans(nf, self.ngpools, self.n2g)
        from_e = apply_poolings_and_trans(ef, self.egpools, self.e2g)
        return features.replace_global_features(
            sum_not_none_elements([from_g, from_n, from_e])
        )

    def post_ops(self, graph: DGLGraph, features: MultiFeatures):
        return features

    __transforms__ = (
        ["g2e", "ns2e", "nd2e", "e2e"] + ["g2n", "n2n", "e2n"] + ["g2g", "n2g", "e2g"]
    )

    @classmethod
    def register_layer_args(cls, parser, prefix=None):
        cls._set_parser_and_prefix(parser, prefix)
        cls._add_argument(
            "layer_bias",
            "-layerbias",
            action="store_true",
            help="use bias for the transforms",
        )
        cls._add_argument(
            "attn_drop", "-attdr", type=float, default=0.0, help="Attn Drop rate"
        )
        cls._add_argument(
            "negative_slope",
            "-negsl",
            type=float,
            default=0.2,
            help="Negative slope of LeakyReLU",
        )
        # Recommend to disable: e2e, nd2e
        cls._add_argument(
            "disables",
            "-dis",
            type=str,
            nargs="*",
            default=["e2e", "nd2e"],
            choices=cls.__transforms__,
            help="disable some transforms",
        )
        # Recommend to copy: e2n
        cls._add_argument(
            "copies",
            "-copy",
            type=str,
            nargs="*",
            default=["e2n"],
            choices=cls.__transforms__,
            help="let some transforms copy the input (when dims match)",
        )
        # The choice of enpool is important in some datasets,
        # E.g. `pseudotree` dataset need to use `sum`.
        # If you use multiple pools, it's better to align
        # edim * len(enpools) and ndim, so that e2n trans can be bypassed.
        cls._add_argument(
            "enpools",
            "-enp",
            type=str,
            default=["sum"],
            choices=["mean", "sum", "max", "att", "gatesum"],
            nargs="+",
            help="Aggregators for edges features to nodes",
        )
        cls._add_argument(
            "ngpools",
            "-ngp",
            type=str,
            default=["mean"],
            choices=["mean", "sum", "max"],
            nargs="*",
            help="Aggregators for nodes features to global",
        )
        cls._add_argument(
            "egpools",
            "-egp",
            type=str,
            default=["mean"],
            choices=["mean", "sum", "max"],
            nargs="*",
            help="Aggregators for edges features to global",
        )
