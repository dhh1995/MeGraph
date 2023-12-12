import torch
import torch.nn.functional as F


class WeightedCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # calculating label weights for weighted loss computation
        n_classes = input.shape[-1]
        V = target.size(0)
        label_count = torch.bincount(target)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(n_classes).long().to(input.device)
        cluster_sizes[torch.unique(target)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes > 0).float()

        # weighted cross-entropy for unbalanced classes
        return F.cross_entropy(
            input,
            target,
            weight=weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )
