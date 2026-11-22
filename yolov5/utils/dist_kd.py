"""copy from https://github.com/hunto/image_classification_sota/blob/d9662f7df68fe46b973c4580b7c9b896cedcd301/lib/models/losses/dist_kd.py"""

import torch.nn as nn
import torch


def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(1).unsqueeze(1), b - b.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


class DIST(nn.Module):
    def __init__(self, beta=1.0, gamma=1.0, tau=1.0):
        super(DIST, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.tau = tau

    def forward(self, preds_S, preds_T):
        assert len(preds_S) == len(preds_T)
        dLoss = []
        dLoss_items = []
        for branchS, branchT in zip(preds_S, preds_T):
            y_s = (branchS / self.tau).softmax(dim=1)
            y_t = (branchT / self.tau).softmax(dim=1)
            inter_loss = self.tau**2 * inter_class_relation(y_s, y_t)
            intra_loss = self.tau**2 * intra_class_relation(y_s, y_t)
            kd_loss = self.beta * inter_loss + self.gamma * intra_loss
            dLoss.append(kd_loss)
            dLoss_items.append(kd_loss)
        dLoss = sum(dLoss) / len(dLoss)
        dLoss_items.append(dLoss)
        dLoss_items = torch.tensor(dLoss_items).to(dLoss.device)
        return dLoss, dLoss_items.detach()
