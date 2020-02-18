import abc
import torch
from torch import nn


class Loss(abc.ABC):
    def __call__(self, lhs, rhs):
        pass


class MSE(Loss):
    """Mean squared error."""

    def __init__(self):
        self._loss = nn.MSELoss()

    def __call__(self, lhs, rhs):
        return self._loss(lhs, rhs)


class GraphMSE(Loss):
    """
    Mean squared error for graphs.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, lhs, rhs):
        return torch.mean(torch.pow(lhs.ndata["feat"] - rhs.ndata["feat"], 2))
