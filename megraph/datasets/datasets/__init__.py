#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : __init__.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
#
# Distributed under terms of the MIT license.

from megraph.io_utils import import_dir_files

from ..manager import DatasetManager
from .base import BaseSyntheticDataset, SyntheticMultiGraphDataset

# Singleton
graph_dataset_manager = DatasetManager()


def register_function(name, meta_data):
    def register_function_fn(fn):
        meta_data["fn"] = fn
        graph_dataset_manager.add_dataset(name, meta_data)
        return fn

    return register_function_fn


import_dir_files(__file__)
