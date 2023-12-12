#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : split.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
#
# Distributed under terms of the MIT license.

import numpy as np
import torch

__all__ = ["get_mask", "get_split_masks", "get_dataset_split"]


def get_mask(n, permutation, start, end):
    mask = np.zeros(n, dtype=bool)
    mask[permutation[start:end]] = True
    return torch.from_numpy(mask)


def get_split_masks(total, split_ratio):
    num_train = int(total * split_ratio["train"])
    num_val = int(total * split_ratio.get("val", 0.0))
    num_test = total - num_train - num_val
    p = np.random.permutation(total)
    train_mask = get_mask(total, p, 0, num_train)
    val_mask = get_mask(total, p, num_train, num_train + num_val)
    test_mask = get_mask(total, p, total - num_test, total)
    return train_mask, val_mask, test_mask


def get_dataset_split(data, split_ratio=None):
    """Get train/val/test split for node classification datasets"""
    if isinstance(data, tuple):
        train_data, val_data, test_data = data
        train_mask = train_data[0].ndata["mask"]
        val_mask = val_data[0].ndata["mask"]
        test_mask = test_data[0].ndata["mask"]
        return train_mask, val_mask, test_mask

    if split_ratio is None:
        # Use pre-defined split
        # Raise error when these mask does not exists in graph ndata.
        g = data[0]
        train_mask = g.ndata["train_mask"].to(torch.bool)
        val_mask = g.ndata["val_mask"].to(torch.bool)
        test_mask = g.ndata["test_mask"].to(torch.bool)
        return train_mask, val_mask, test_mask
    # Random
    if type(split_ratio) is not dict:
        raise ValueError(f"split ratio must be a dict")
    if len(data) > 1:
        total = len(data)
    else:
        g = data[0]
        if type(g) is tuple:
            g = g[0]
        total = g.num_nodes()
    return get_split_masks(total, split_ratio)
