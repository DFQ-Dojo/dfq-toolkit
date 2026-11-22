import torch.nn as nn
import torch.nn.functional as F
import torch

"""copy from https://github.com/hunto/image_classification_sota/blob/d9662f7df68fe46b973c4580b7c9b896cedcd301/lib/models/losses/kl_div.py#L5"""


class KLDivergence(nn.Module):
    """A measure of how one probability distribution Q is different from a
    second, reference probability distribution P.

    Args:
        tau (float): Temperature coefficient. Defaults to 1.0.
        reduction (str): Specifies the reduction to apply to the loss:
            ``'none'`` | ``'batchmean'`` | ``'sum'`` | ``'mean'``.
            ``'none'``: no reduction will be applied,
            ``'batchmean'``: the sum of the output will be divided by
                the batchsize,
            ``'sum'``: the output will be summed,
            ``'mean'``: the output will be divided by the number of
                elements in the output.
            Default: ``'batchmean'``
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(
        self,
        tau=1.0,
        reduction="batchmean",
    ):
        super(KLDivergence, self).__init__()
        self.tau = tau

        accept_reduction = {"none", "batchmean", "sum", "mean"}
        assert reduction in accept_reduction, (
            f"KLDivergence supports reduction {accept_reduction}, but gets {reduction}."
        )
        self.reduction = reduction

    def forward(self, preds_S, preds_T):
        """Forward computation.

        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W) or shape (N, C).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W) or shape (N, C).

        Return:
            torch.Tensor: The calculated loss value.
        """
        assert len(preds_S) == len(preds_T)
        dLoss = []
        dLoss_items = []
        for branchS, branchT in zip(preds_S, preds_T):
            softmax_pred_T = F.softmax(branchT / self.tau, dim=1)
            logsoftmax_preds_S = F.log_softmax(branchS / self.tau, dim=1)
            loss = (self.tau**2) * F.kl_div(logsoftmax_preds_S, softmax_pred_T, reduction=self.reduction)
            dLoss.append(loss)
            dLoss_items.append(loss)
        dLoss = sum(dLoss)
        dLoss_items.append(dLoss)
        dLoss_items = torch.tensor(dLoss_items).to(dLoss.device)
        return dLoss, dLoss_items.detach()
