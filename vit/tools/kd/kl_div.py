from typing import List
import torch.nn as nn
import torch.nn.functional as F
import torch

"""copy from https://github.com/hunto/image_classification_sota/blob/d9662f7df68fe46b973c4580b7c9b896cedcd301/lib/models/losses/kl_div.py#L5"""


class KLDivergence(nn.Module):
    """A measure of how one probability distribution Q is different from a
    second, reference probability distribution P.

    Args:
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

    def __init__(self, reduction="batchmean"):

        super(KLDivergence, self).__init__()

        accept_reduction = {"none", "batchmean", "sum", "mean"}
        assert reduction in accept_reduction, (
            f"KLDivergence supports reduction {accept_reduction}, "
            f"but gets {reduction}."
        )
        self.reduction = reduction

    def forward(
        self, preds_S: List[torch.Tensor], preds_T: List[torch.Tensor]
    ) -> torch.Tensor:
        """Forward computation.

        Args:
            preds_S (torch.Tensor): The student model prediction with a list of
                shape (N, C, H, W).
            preds_T (torch.Tensor): The teacher model prediction with a list of
                shape (N, C, H, W).

        Return:
            torch.Tensor: The calculated loss value.
        """
        assert len(preds_S) == len(preds_T)
        assert all(s.shape == t.shape for s, t in zip(preds_S, preds_T))

        def flatten(tensors: List[torch.Tensor]) -> torch.Tensor:
            flattened_tensors = [t.view(t.shape[0], -1) for t in tensors]
            return torch.cat(flattened_tensors, dim=1)

        pred_t = flatten(preds_T).detach()
        pred_s = flatten(preds_S)

        softmax_pred_T = F.softmax(pred_t, dim=1)
        logsoftmax_preds_S = F.log_softmax(pred_s, dim=1)
        return F.kl_div(
            logsoftmax_preds_S, softmax_pred_T, reduction=self.reduction
        )
