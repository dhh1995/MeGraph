# From https://github.com/vandit15/Class-balanced-loss-pytorch/blob/master/class_balanced_loss.py

import numpy as np
import torch
import torch.nn.functional as F


def focal_loss(labels: torch.Tensor, logits: torch.Tensor, alpha: float, gamma: float):
    """Compute the focal loss between `logits` and the ground truth `labels`.
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.
    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    BCLoss = F.binary_cross_entropy_with_logits(
        input=logits, target=labels, reduction="none"
    )

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(
            -gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits))
        )

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss


class CBLoss(torch.nn.Module):
    r"""Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.
    Args:
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.
    """

    def __init__(
        self,
        samples_per_cls: list,
        no_of_classes: int,
        beta: float = 0.9999,
        gamma: float = 2.0,
        loss_type: str = "focal",
    ):
        super().__init__()
        assert loss_type in [
            "focal",
            "sigmoid",
            "softmax",
        ], f"No loss type {loss_type} in CBLoss"

        self.samples_per_cls = samples_per_cls
        self.no_of_classes = no_of_classes
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type

        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - self.beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * no_of_classes
        weights = torch.tensor(weights).float()
        weights = weights.unsqueeze(0)

        self.weights = weights

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        labels_one_hot = F.one_hot(labels, self.no_of_classes).float()
        weights = self.compute_weights(labels_one_hot)
        if self.loss_type == "focal":
            cb_loss = focal_loss(labels_one_hot, logits, weights, self.gamma)
        elif self.loss_type == "sigmoid":
            cb_loss = F.binary_cross_entropy_with_logits(
                input=logits, target=labels_one_hot, weights=weights
            )
        elif self.loss_type == "softmax":
            pred = logits.softmax(dim=1)
            cb_loss = F.binary_cross_entropy(
                input=pred, target=labels_one_hot, weight=weights
            )
        return cb_loss

    def compute_weights(self, labels_one_hot: torch.Tensor):
        weights = self.weights.to(labels_one_hot.device)
        weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
        weights = weights.sum(dim=1, keepdim=True)
        weights = weights.repeat(1, self.no_of_classes)
        return weights
