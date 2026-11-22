import torch.nn as nn
import torch


class MSE(nn.Module):
    """
    MSE between student and teacher networks.

    Precondition: student and teacher networks are required to be the SAME ARCHITECTURE!!
    """

    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, preds_S, preds_T):
        assert len(preds_S) == len(preds_T)
        dLoss = []
        for branchS, branchT in zip(preds_S, preds_T):
            dLoss.append(torch.mean((branchS - branchT) ** 2))

        dLoss = sum(dLoss) / len(dLoss)
        dLoss_items = torch.tensor((0.0, 0.0, 0.0, dLoss.item())).to(
            dLoss.device
        )
        return dLoss, dLoss_items.detach()
