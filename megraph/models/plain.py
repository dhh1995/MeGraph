#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : plain.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
#
# Distributed under terms of the MIT license.

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import register_function
from .model import MultiFeaturesModel


@register_function("plain")
class PlainModel(MultiFeaturesModel):
    def __init__(
        self,
        input_dims,
        output_dims,
        pe_dim,
        task,
        build_conv,
        stem_beta,
        branch_beta,
        soft_readout=False,
        # MultiFeaturesModel Args
        **kwargs,
    ):
        super(PlainModel, self).__init__(
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
        current_dims = self.get_input_dims_after_encoder()
        self.layers = nn.ModuleList()
        for i in range(self.num_layers + 1):
            conv = self.get_conv(i, current_dims, self.hidden_dims)
            self.layers.append(self.get_conv_block(conv))
            current_dims = conv.get_output_dims()
        self.prepare_last_layer(current_dims)
        self.weights = torch.nn.Parameter(torch.randn((len(self.layers) - 1)))
        torch.nn.init.normal_(self.weights)

    def forward(self, graph: dgl.DGLGraph):
        features = self.get_inputs(graph)
        outs = []
        for i, layer in enumerate(self.layers):
            features = features.residual_when_same_shape(
                layer(graph, features),
                stem_beta=self.stem_beta,
                branch_beta=self.branch_beta,
            )
            features = self.apply_post_layer_oprs(features, ind=i)
            if i > 0:
                outs.append(features)
        if self.soft_readout:
            weights = F.softmax(self.weights)
            features = sum([o * w for o, w in zip(outs, weights)])
        logits = self.apply_last_layer(graph, features)
        return logits

    @classmethod
    def register_model_args(cls, parser, prefix=None):
        super().register_model_args(parser, prefix=prefix)
        cls._add_argument(
            "branch_beta",
            "-bb",
            type=float,
            default=0.5,
            help="Beta for Branch in residual",
        )
        cls._add_argument(
            "stem_beta",
            "-sb",
            type=float,
            default=1.0,
            help="Beta for Stem in residual",
        )
        cls._add_argument(
            "soft_readout", "-sr", action="store_true", help="readout use softmax"
        )
