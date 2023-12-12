#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : evaluator.py
# Author : Honghua Dong
# Email  : dhh19951@gmail.com
#
# Distributed under terms of the MIT license.

import numpy as np
import torch
from ogb.graphproppred import Evaluator as GPredEvaluator
from ogb.linkproppred import Evaluator as LPredEvaluator
from ogb.nodeproppred import Evaluator as NPredEvaluator
from sklearn.metrics import confusion_matrix, f1_score
from torchmetrics.functional import average_precision

OGB_EVALUATORS = dict(
    npred=NPredEvaluator,
    lpred=LPredEvaluator,
    gpred=GPredEvaluator,
)


class BaseEvaluator(object):
    def __init__(self):
        super().__init__()

    def _get_results(self, pred, labels):
        """get evaluation results from pred and labels"""
        raise NotImplementedError()

    def eval(self, input_dict):
        """Evaluate the prediction results (y_pred) with the ground truth (y_true)"""
        pred = input_dict["y_pred"]
        labels = input_dict["y_true"]
        if pred.shape != labels.shape:
            raise ValueError(
                f"y_pred and y_true should have the same shape, "
                f"got {pred.shape} and {labels.shape}"
            )
        return self._get_results(pred, labels)


class AccEvaluator(BaseEvaluator):
    """Evaluator for accuracy"""

    def __init__(self, dataset_name):
        super().__init__()
        if dataset_name in ["sbm_cluster", "sbm_pattern"]:
            self._get_results = getattr(self, "_get_SBM_acc")
        else:
            self._get_results = getattr(self, "_get_acc")

    def _get_acc(self, pred, labels):
        matched = (pred == labels).float()
        return dict(acc=torch.mean(matched).item())

    def _get_SBM_acc(self, pred, labels):
        S = labels.cpu().detach().numpy()
        C = pred.cpu().detach().numpy()
        CM = confusion_matrix(S, C).astype(np.float32)
        nb_classes = CM.shape[0]
        nb_non_empty_classes = 0
        pr_classes = np.zeros(nb_classes)
        for r in range(nb_classes):
            cluster = np.where(S == r)[0]
            if cluster.shape[0] != 0:
                pr_classes[r] = CM[r, r] / float(cluster.shape[0])
                if CM[r, r] > 0:
                    nb_non_empty_classes += 1
            else:
                pr_classes[r] = 0.0
        acc = 100.0 * np.sum(pr_classes) / float(nb_classes)
        return dict(SBM_acc=acc)


class ClassBalanceEvaluator(BaseEvaluator):
    """Evaluator for class-balanced accuracy"""

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def _get_results(self, pred, labels):
        acc = []
        for i in range(self.num_classes):
            i_indexs = torch.where(labels == i)
            if i_indexs[0].shape[0] == 0:
                continue
            i_pred = pred[i_indexs]
            i_acc = torch.mean((i_pred == i).float()).item()
            acc.append(i_acc)
        return dict(acc=np.mean(acc))


class RegEvaluator(BaseEvaluator):
    """Regression evaluator (L2 and MAE)"""

    def __init__(self, eval_metric="L2"):
        super().__init__()
        self._get_results = getattr(self, f"_get_{eval_metric}_results")

    def _get_L2_results(self, pred, labels):
        res = (pred - labels).pow(2)
        return dict(L2=torch.mean(res).item())

    def _get_MAE_results(self, pred, labels):
        res = torch.nn.functional.l1_loss(pred, labels)
        return dict(MAE=res.detach().item())


class F1Evaluator(BaseEvaluator):
    """evaluator for f1 score"""

    def __init__(self):
        super().__init__()

    def _get_results(self, pred, labels):
        labels = labels.cpu().detach().numpy()
        pred = pred.cpu().detach().numpy()
        f1 = f1_score(labels, pred, average="macro", zero_division=0)
        return dict(f1=f1)


class APEvaluator(BaseEvaluator):
    """evaluator for average precision (AP)"""

    def __init__(self):
        super().__init__()

    def _get_results(self, pred, labels):
        """
        compute Average Precision (AP) averaged across tasks
        """
        labels_nans = torch.isnan(labels)
        labels_list = [
            labels[..., ii][~labels_nans[..., ii]] for ii in range(labels.shape[-1])
        ]
        pred_list = [
            pred[..., ii][~labels_nans[..., ii]] for ii in range(pred.shape[-1])
        ]
        metric_val = []
        for ii in range(len(labels_list)):
            res = average_precision(pred_list[ii], labels_list[ii].int(), pos_label=1)
            metric_val.append(res)
        x = torch.stack(metric_val)
        ap = torch.div(torch.nansum(x), (~torch.isnan(x)).count_nonzero())
        return dict(ap=ap.detach().item())


def get_evaluator(name, task, use_cb_eval=False, num_classes=None, meta_data={}):
    if use_cb_eval:
        return ClassBalanceEvaluator(num_classes)
    if meta_data.get("reg", False):
        return RegEvaluator(meta_data["eval_metric"])
    if name.startswith("ogb"):
        return OGB_EVALUATORS[task](name)
    if meta_data.get("eval_metric") == "f1":
        return F1Evaluator()
    if meta_data.get("eval_metric") == "ap":
        return APEvaluator()
    return AccEvaluator(name)
