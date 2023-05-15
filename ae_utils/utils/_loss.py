from abc import abstractmethod
from typing import Optional
import warnings

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ae_utils.utils.distances import DistanceFunction, CosineDistance, PNormDistance
from ae_utils.utils.kernels import KernelFunction
from ae_utils.utils.registry import ClassRegistry

LossRegistry = ClassRegistry()

warnings.warn("The `LossFunction` module is deprecated!", DeprecationWarning)


class LossFunction(nn.Module):
    @abstractmethod
    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        pass


@LossRegistry.register("cont")
class ContrastiveLoss(LossFunction):
    def __init__(self, df_x: DistanceFunction | None = None, df_y: DistanceFunction | None = None):
        super().__init__()

        self.df_x = df_x or CosineDistance()
        self.df_y = df_y or PNormDistance(torch.inf)

    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        return F.mse_loss(self.df_x(X, X), self.df_y(Y, Y), reduction="mean")


@LossRegistry.register("mmd")
class MMDLoss(LossFunction):
    def __init__(self, kernel: KernelFunction):
        super().__init__()

        self.kernel = kernel

    def forward(self, X: Tensor, Y: Tensor) -> Tensor:
        K_xx = self.kernel(X, X)
        K_yy = self.kernel(Y, Y)
        K_xy = self.kernel(X, Y)

        return self.tril_mean(K_xx) + self.tril_mean(K_yy) - 2 * K_xy.mean()

    @staticmethod
    def tril_mean(A: Tensor, offset: int = -1):
        """The mean of the lower triangular (given the input offset) elements in the tensor `A`"""
        idxs = torch.tril_indices(*A.shape, offset).unbind()

        return A[idxs].mean()
