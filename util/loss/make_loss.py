# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth

def make_loss(num_classes, ignore_index=-1):

    xent = CrossEntropyLabelSmooth(num_classes=num_classes)
    print("label smooth on, numclasses:", num_classes)

    def loss_func(i2tscore, target):

        mask = target!= ignore_index
        i2tscore = i2tscore[mask]
        target = target[mask]

        if mask.sum() == 0:
            return torch.tensor([0.0]).cuda()

        I2TLOSS = xent(i2tscore, target)

        return I2TLOSS

    return loss_func
class DCLLoss(nn.Module):
    def __init__(self, sigma=1, delta=1):
        super(DCLLoss, self).__init__()
        self.sigma = sigma
        self.delta = delta

    def forward(self, s_emb, t_emb):
        N = len(s_emb)
        S_dist = torch.cdist(s_emb, s_emb)
        S_dist = S_dist / S_dist.mean(1, keepdim=True)

        with torch.no_grad():
            T_dist = torch.cdist(t_emb, t_emb)
            W = torch.exp(-T_dist.pow(2) / self.sigma)

            identity_matrix = torch.eye(N).cuda(non_blocking=True)
            pos_weight = (W) * (1 - identity_matrix)
            neg_weight = (1 - W) * (1 - identity_matrix)

        pull_losses = torch.relu(S_dist).pow(2) * pos_weight
        push_losses = torch.relu(self.delta - S_dist).pow(2) * neg_weight

        loss = (pull_losses.sum() + push_losses.sum()) / (len(s_emb) * (len(s_emb) - 1))

        return loss

class RCLLoss(nn.Module):
    def __init__(self, sigma=1, delta=1):
        super(RCLLoss, self).__init__()
        self.sigma = sigma
        self.delta = delta

    def forward(self, s_emb, t_emb):
        N = len(s_emb)
        S_dist = torch.cdist(s_emb, s_emb)
        S_dist = S_dist / S_dist.mean(1, keepdim=True)

        with torch.no_grad():
            T_dist = torch.cdist(t_emb, t_emb)
            W = torch.exp(-T_dist.pow(2) / self.sigma)

            identity_matrix = torch.eye(N).cuda(non_blocking=True)
            pos_weight = (W) * (1 - identity_matrix)
            neg_weight = (1 - W) * (1 - identity_matrix)

        pull_losses = torch.relu(S_dist).pow(2) * pos_weight
        push_losses = torch.relu(self.delta - S_dist).pow(2) * neg_weight

        loss = (pull_losses.sum() + push_losses.sum()) / (len(s_emb) * (len(s_emb) - 1))

        return loss
