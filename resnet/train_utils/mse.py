import torch.nn as nn
import torch


class MSE(nn.Module):
    def __init__(
        self,
    ):
        super(MSE, self).__init__()

    def forward(self, preds_S, preds_T, debug_S=None, debug_T=None):
        """
        mse between predT & predS
        only works when Stu & Tea are same architecture
        """
        assert len(preds_S) == len(preds_T)
        dLoss = []
        for idx, (branchS, branchT) in enumerate(zip(preds_S, preds_T)):
            if branchS.shape != branchT.shape:
                # print('student:', debug_S[idx])
                # print('teacher:', debug_T[idx])
                # print()
                continue
            dLoss.append(torch.mean((branchS - branchT) ** 2))
        dLoss = sum(dLoss) / len(dLoss)
        dLoss_items = torch.tensor((0.0, 0.0, 0.0, dLoss.item())).to(
            dLoss.device
        )
        return dLoss, dLoss_items.detach()
