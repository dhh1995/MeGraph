#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : models.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
#
# Distributed under terms of the MIT license.

"""
Model zoo for graph models
Modified based on https://github.com/dmlc/dgl/tree/master/examples/pytorch/model_zoo/citation_network
"""

import torch
import torch.nn as nn
from dgl.nn.pytorch import (AGNNConv, APPNPConv, ChebConv, GATv2Conv, GINConv,
                            GraphConv, SAGEConv, SGConv, TAGConv)
from megraph.torch_utils import get_activation

from . import register_function
from .model import GraphModel


@register_function("gcn")
class GCN(GraphModel):
    def __init__(
        self,
        in_feats,
        n_classes,
        n_hidden,
        n_layers,
        activation,
        dropout,
        enable_bn=True,
    ):
        super(GCN, self).__init__()
        self.activation = get_activation(activation)
        self.enable_bn = enable_bn
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden))
        self.bns.append(torch.nn.BatchNorm1d(n_hidden))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden))
            self.bns.append(torch.nn.BatchNorm1d(n_hidden))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, graph):
        h = graph.ndata["feat"]
        for i, layer in enumerate(self.layers):
            if i != 0:
                if self.enable_bn:
                    h = self.bns[i - 1](h)
                h = self.activation(h)
                h = self.dropout(h)
            h = layer(graph, h)
        return h

    @classmethod
    def register_model_args(cls, parser, prefix=None):
        cls._set_parser_and_prefix(parser, prefix)
        cls._add_argument("n_hidden", "-nhd", type=int, default=16, help="Hidden dim")
        cls._add_argument(
            "n_layers", "-nl", type=int, default=1, help="Number of layers"
        )
        cls._add_argument(
            "activation", "-act", type=str, default="relu", help="Activation"
        )
        cls._add_argument("dropout", "-drop", type=float, default=0.5, help="Drop rate")
        cls._add_argument(
            "enable_bn", "-bn", action="store_true", help="Use batch norm"
        )


@register_function("gat")
class GAT(GraphModel):
    def __init__(
        self,
        in_feats,
        n_classes,
        n_hidden,
        n_layers,
        heads,
        activation,
        feat_drop,
        attn_drop,
        negative_slope,
        residual,
    ):
        super(GAT, self).__init__()
        self.n_layers = n_layers
        self.gat_layers = nn.ModuleList()
        self.activation = get_activation(activation)
        # input projection (no residual)
        self.gat_layers.append(
            GATv2Conv(
                in_feats,
                n_hidden,
                heads[0],
                feat_drop,
                attn_drop,
                negative_slope,
                False,
                self.activation,
            )
        )
        # hidden layers
        for l in range(1, n_layers):
            # due to multi-head, the in_dim = n_hidden * num_heads
            self.gat_layers.append(
                GATv2Conv(
                    n_hidden * heads[l - 1],
                    n_hidden,
                    heads[l],
                    feat_drop,
                    attn_drop,
                    negative_slope,
                    residual,
                    self.activation,
                )
            )
        # output projection
        self.gat_layers.append(
            GATv2Conv(
                n_hidden * heads[-2],
                n_classes,
                heads[-1],
                feat_drop,
                attn_drop,
                negative_slope,
                residual,
                None,
            )
        )

    def forward(self, graph):
        h = graph.ndata["feat"]
        for l in range(self.n_layers):
            h = self.gat_layers[l](graph, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](graph, h).mean(1)
        return logits

    @classmethod
    def register_model_args(cls, parser, prefix=None):
        cls._set_parser_and_prefix(parser, prefix)
        cls._add_argument("n_hidden", type=int, default=8, help="Hidden dim")
        cls._add_argument("n_layers", type=int, default=1, help="Number of layers")
        cls._add_argument("activation", type=str, default="elu", help="Activation")
        cls._add_argument("residual", action="store_true", help="Residual link")
        cls._add_argument("feat_drop", type=float, default=0.6, help="Feat Drop rate")
        cls._add_argument("attn_drop", type=float, default=0.6, help="Attn Drop rate")
        cls._add_argument(
            "heads", type=int, nargs="+", default=[8] * 1 + [1], help="Attn Head num"
        )
        cls._add_argument(
            "negative_slope", type=float, default=0.2, help="Negative slope of elu"
        )


@register_function("sage")
class GraphSAGE(GraphModel):
    def __init__(
        self,
        in_feats,
        n_classes,
        n_hidden,
        n_layers,
        activation,
        dropout,
        aggregator_type,
    ):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        activation = get_activation(activation)

        # input layer
        self.layers.append(
            SAGEConv(
                in_feats,
                n_hidden,
                aggregator_type,
                feat_drop=dropout,
                activation=activation,
            )
        )
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(
                SAGEConv(
                    n_hidden,
                    n_hidden,
                    aggregator_type,
                    feat_drop=dropout,
                    activation=activation,
                )
            )
        # output layer
        self.layers.append(
            SAGEConv(
                n_hidden, n_classes, aggregator_type, feat_drop=dropout, activation=None
            )
        )  # activation None

    def forward(self, graph):
        h = graph.ndata["feat"]
        for layer in self.layers:
            h = layer(graph, h)
        return h

    @classmethod
    def register_model_args(cls, parser, prefix=None):
        cls._set_parser_and_prefix(parser, prefix)
        cls._add_argument("n_hidden", type=int, default=16, help="Hidden dim")
        cls._add_argument("n_layers", type=int, default=1, help="Number of layers")
        cls._add_argument("activation", type=str, default="relu", help="Activation")
        cls._add_argument("dropout", type=float, default=0.5, help="Drop rate")
        cls._add_argument("aggregator_type", type=str, default="gcn", help="Aggregator")


@register_function("appnp")
class APPNP(GraphModel):
    def __init__(
        self,
        in_feats,
        n_classes,
        n_hidden,
        n_layers,
        activation,
        feat_drop,
        edge_drop,
        alpha,
        k,
    ):
        super(APPNP, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(nn.Linear(in_feats, n_hidden))
        # hidden layers
        for i in range(1, n_layers):
            self.layers.append(nn.Linear(n_hidden, n_hidden))
        # output layer
        self.layers.append(nn.Linear(n_hidden, n_classes))
        self.activation = get_activation(activation)
        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        self.propagate = APPNPConv(k, alpha, edge_drop)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, graph):
        # prediction step
        h = graph.ndata["feat"]
        h = self.feat_drop(h)
        h = self.activation(self.layers[0](h))
        for layer in self.layers[1:-1]:
            h = self.activation(layer(h))
        h = self.layers[-1](self.feat_drop(h))
        # propagation step
        h = self.propagate(graph, h)
        return h

    @classmethod
    def register_model_args(cls, parser, prefix=None):
        cls._set_parser_and_prefix(parser, prefix)
        cls._add_argument("n_hidden", type=int, default=64, help="Hidden dim")
        cls._add_argument("n_layers", type=int, default=1, help="Number of layers")
        cls._add_argument("activation", type=str, default="relu", help="Activation")
        cls._add_argument("feat_drop", type=float, default=0.5, help="Feat Drop rate")
        cls._add_argument("edge_drop", type=float, default=0.5, help="Edge Drop rate")
        cls._add_argument("alpha", type=float, default=0.1, help="Alpha")
        cls._add_argument("k", type=float, default=10, help="K")


@register_function("tagcn")
class TAGCN(GraphModel):
    def __init__(self, in_feats, n_classes, n_hidden, n_layers, activation, dropout):
        super(TAGCN, self).__init__()
        activation = get_activation(activation)
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(TAGConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(TAGConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(TAGConv(n_hidden, n_classes))  # activation=None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, graph):
        h = graph.ndata["feat"]
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(graph, h)
        return h

    @classmethod
    def register_model_args(cls, parser, prefix=None):
        cls._set_parser_and_prefix(parser, prefix)
        cls._add_argument("n_hidden", type=int, default=16, help="Hidden dim")
        cls._add_argument("n_layers", type=int, default=1, help="Number of layers")
        cls._add_argument("activation", type=str, default="relu", help="Activation")
        cls._add_argument("dropout", type=float, default=0.5, help="Drop rate")


@register_function("agnn")
class AGNN(GraphModel):
    def __init__(
        self, in_feats, n_classes, n_hidden, n_layers, init_beta, learn_beta, dropout
    ):
        super(AGNN, self).__init__()
        self.layers = nn.ModuleList(
            [AGNNConv(init_beta, learn_beta) for _ in range(n_layers)]
        )
        self.proj = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_feats, n_hidden), nn.ReLU()
        )
        self.cls = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_hidden, n_classes))

    def forward(self, graph):
        h = self.proj(graph.ndata["feat"])
        for layer in self.layers:
            h = layer(graph, h)
        return self.cls(h)

    @classmethod
    def register_model_args(cls, parser, prefix=None):
        cls._set_parser_and_prefix(parser, prefix)
        cls._add_argument("n_hidden", type=int, default=32, help="Hidden dim")
        cls._add_argument("n_layers", type=int, default=2, help="Number of layers")
        cls._add_argument("dropout", type=float, default=0.5, help="Drop rate")
        cls._add_argument("init_beta", type=float, default=1.0, help="Initial Beta")
        cls._add_argument("learn_beta", action="store_false", help="Learn Beta")


@register_function("sgc")
class SGC(GraphModel):
    def __init__(self, in_feats, n_classes, n_hidden, k, bias):
        super(SGC, self).__init__()
        self.net = SGConv(in_feats, n_classes, k=k, cached=True, bias=bias)

    def forward(self, graph):
        h = graph.ndata["feat"]
        return self.net(graph, h)

    @classmethod
    def register_model_args(cls, parser, prefix=None):
        cls._set_parser_and_prefix(parser, prefix)
        cls._add_argument("n_hidden", type=int, default=None, help="Hidden dim")
        cls._add_argument("k", type=float, default=2, help="K")
        cls._add_argument("bias", type=bool, default=False, help="Use bias")


@register_function("gin")
class GIN(GraphModel):
    def __init__(
        self, in_feats, n_classes, n_hidden, n_layers, dropout, init_eps, learn_eps
    ):
        super(GIN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            GINConv(
                nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(in_feats, n_hidden),
                    nn.ReLU(),
                ),
                "mean",
                init_eps,
                learn_eps,
            )
        )
        for i in range(n_layers - 1):
            self.layers.append(
                GINConv(
                    nn.Sequential(
                        nn.Dropout(dropout), nn.Linear(n_hidden, n_hidden), nn.ReLU()
                    ),
                    "mean",
                    init_eps,
                    learn_eps,
                )
            )
        self.layers.append(
            GINConv(
                nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(n_hidden, n_classes),
                ),
                "mean",
                init_eps,
                learn_eps,
            )
        )

    def forward(self, graph):
        h = graph.ndata["feat"]
        for layer in self.layers:
            h = layer(graph, h)
        return h

    @classmethod
    def register_model_args(cls, parser, prefix=None):
        cls._set_parser_and_prefix(parser, prefix)
        cls._add_argument("n_hidden", type=int, default=16, help="Hidden dim")
        cls._add_argument("n_layers", type=int, default=1, help="Number of layers")
        cls._add_argument("dropout", type=float, default=0.6, help="Drop rate")
        cls._add_argument("init_eps", type=float, default=0, help="Initial Eps")
        cls._add_argument("learn_eps", action="store_false", help="Learn Eps")


@register_function("chebnet")
class ChebNet(GraphModel):
    def __init__(self, in_feats, n_classes, n_hidden, n_layers, k, bias):
        super(ChebNet, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(ChebConv(in_feats, n_hidden, k, bias=bias))
        for _ in range(n_layers - 1):
            self.layers.append(ChebConv(n_hidden, n_hidden, k, bias=bias))

        self.layers.append(ChebConv(n_hidden, n_classes, k, bias=bias))

    def forward(self, graph):
        h = graph.ndata["feat"]
        for layer in self.layers:
            h = layer(graph, h, [2])
        return h

    @classmethod
    def register_model_args(cls, parser, prefix=None):
        cls._set_parser_and_prefix(parser, prefix)
        cls._add_argument("n_hidden", type=int, default=32, help="Hidden dim")
        cls._add_argument("n_layers", type=int, default=1, help="Number of layers")
        cls._add_argument("k", type=float, default=2, help="K")
        cls._add_argument("bias", action="store_false", help="Use bias")


@register_function("cheat")
class CheatShortest(GraphModel):
    def __init__(self, in_feats, n_classes):
        super(CheatShortest, self).__init__()
        self.in_feats = in_feats
        self.n_classes = n_classes
        self.scale = nn.Parameter(torch.empty(1))
        nn.init.ones_(self.scale)

    def forward(self, graph):
        h = graph.ndata["feat"]
        res = []
        for i in range(self.n_classes):
            indices0 = h[:, i * 2 + 0].nonzero()
            indices1 = h[:, i * 2 + 1].nonzero()
            # print(indices0, indices1)
            res.append((indices0 - indices1).abs())
        res = torch.cat(res, dim=1)
        return res * self.scale
