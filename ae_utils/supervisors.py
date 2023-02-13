from abc import abstractmethod
from typing import Optional

import torch
from torch import Tensor, nn
from ae_utils.utils.distances import DistanceFunction

from ae_utils.utils.loss import ContrastiveLoss
from ae_utils.utils.utils import build_ffn


class Supervisor(nn.Module):
    @abstractmethod
    def forward(self, Z: Tensor, Y: Tensor) -> Tensor:
        pass


class DummySupervisor(Supervisor):
    def forward(self, Z: Tensor, Y: Tensor) -> Tensor:
        return torch.tensor(0.0)


class RegressionSupervisor(Supervisor):
    def __init__(self, g_d_h: int = 64, g_n_layers: int = 0, n_tasks: int = 1):
        super().__init__()

        self.ffn = build_ffn(self.d_z, [g_d_h] * g_n_layers, n_tasks)
        self.mse_metric = nn.MSELoss(reduction="mean")

    def forward(self, Z: Tensor, Y: Tensor) -> Tensor:
        mask = ~Y.isnan()
        if mask.any():
            Y_pred = self.ffn(Z)
            return self.mse_metric(Y_pred[mask], Y[mask])

        return torch.tensor(0.0)


class ContrastiveSupervisor(Supervisor):
    def __init__(
        self, df_x: Optional[DistanceFunction] = None, df_y: Optional[DistanceFunction] = None
    ):
        super().__init__()

        self.cont_metric = ContrastiveLoss(df_x, df_y)

    def forward(self, Z: Tensor, Y: Tensor) -> Tensor:
        mask = ~Y.isnan().any(1)

        return self.cont_metric(Z[mask], Y[mask]) if mask.any() else torch.tensor(0.0)
