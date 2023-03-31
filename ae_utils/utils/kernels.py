from abc import abstractmethod

import torch
from torch import Tensor, nn


class KernelFunction(nn.Module):
    @abstractmethod
    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        """
        Parameters
        ----------
        X : Tensor
            an `n x d` tensor
        Y : Tensor
            and `m x d` tensor

        Returns
        -------
        Tensor
            an `n x m` tensor of the distance matrix `K`, where `K[i, j]` is the value of the
            kernel function `k(X[i], Y[j])`
        """


class InverseMultiQuadraticKernel(KernelFunction):
    def __init__(self, c: float = 1):
        super().__init__()
        
        self.c = c

    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        L2_sq = torch.cdist(X, Y) ** 2

        return self.c / (self.c + L2_sq)


IMQKernel = InverseMultiQuadraticKernel
