from abc import abstractmethod

import torch
from torch import Tensor, nn

from ae_utils.utils.registry import ClassRegistry

DistanceFunctionRegistry = ClassRegistry()


class DistanceFunction(nn.Module):
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
            an `n x m` tensor of the distance matrix `D`, where `D[i, j]` is the value of the
            distance function `d(X[i], Y[j])`
        """


@DistanceFunctionRegistry.register("cosine")
class CosineDistance(DistanceFunction):
    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        x_norm = X.norm(dim=1, keepdim=True)
        y_norm = Y.norm(dim=1, keepdim=True)
        S = X @ Y.T / (x_norm @ y_norm.T)

        return 1 - S


@DistanceFunctionRegistry.register("pnorm")
class PNormDistance(DistanceFunction):
    def __init__(self, p: int = 2):
        super().__init__()

        self.p = p
    
    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        return torch.cdist(X, Y, self.p)