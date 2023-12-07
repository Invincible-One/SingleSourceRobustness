import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F



def hinge_loss(yhat, y):
    torch_loss = nn.MarginRankingLoss(margin=1.0, reduction="none")
    y = (y.float() * 2.0) - 1.0
    return torch_loss(yhat[:, 1], yhat[:, 0], y)



class LossComputer:
    def __init__(self, criterion, n_groups, get_group_repr_fn):
        self.criterion = criterion

        self.n_groups = n_groups

        self.get_group_repr_fn = get_group_repr_fn

        self.reset_stats()

    def loss(self, output, y):
        per_sample_loss = self.criterion(output, y)
        loss_v = per_sample_loss.mean()

        return loss_v
    
    def reset_stats(self):
        self.correct = 0
        self.total = 0

        self.group_correct = [0] * self.n_groups
        self.group_total = [0] * self.n_groups

    def update_stats(self, output, y, g):
        predicted = output.argmax(dim=1)
        correct_mask = (predicted == y)
        self.correct += correct_mask.sum().item()
        self.total += y.size(0)

        for i in range(self.n_groups):
            mask = (g == i)
            self.group_correct[i] += (output[mask].argmax(dim=1) == y[mask]).sum().item()
            self.group_total[i] += mask.sum().item()

    def display(self, train):
        acc = self.correct / self.total
        
        group_acc = list()
        group_repr = list()
        for i in range(self.n_groups):
            group_acc.append(self.group_correct[i] / self.group_total[i])
            group_repr.append(self.get_group_repr_fn(i))
        
        if train:
            return f"TRAIN {acc:.3f}"
        else:
            group_acc = list()
            group_repr = list()
            for i in range(self.n_groups):
                group_acc.append(self.group_correct[i] / self.group_total[i])
                group_repr.append(self.get_group_repr_fn(i))
            return f"VAL {acc:.4f} group " + "|".join([f"{group_repr[i]} {group_acc[i]:.4f}" for i in range(self.n_groups)])
