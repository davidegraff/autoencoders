from abc import abstractmethod
from typing import Optional

import torch
from torch import Tensor, nn

from ae_utils.utils import Configurable, DistanceFunction, ContrastiveLoss, build_ffn


class Supervisor(nn.Module, Configurable):
    @abstractmethod
    def forward(self, Z: Tensor, Y: Tensor) -> Tensor:
        pass


class DummySupervisor(Supervisor):
    def forward(self, Z: Tensor, Y: Tensor) -> Tensor:
        return torch.tensor(0.0)

    def to_config(self) -> dict:
        return {}


class RegressionSupervisor(Supervisor):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Optional[int] = None):
        super().__init__()

        self.ffn = build_ffn(input_dim, output_dim, hidden_dims)
        self.mse_metric = nn.MSELoss(reduction="mean")

    def forward(self, Z: Tensor, Y: Tensor) -> Tensor:
        mask = Y.isfinite()
        if mask.any():
            Y_pred = self.ffn(Z)
            return self.mse_metric(Y_pred[mask], Y[mask])

        return torch.tensor(0.0)

    def to_config(self) -> dict:
        hidden_dims = [layer.out_features for layer in self.ffn[-1]] if len(self.ffn) > 1 else None

        return {
            "input_dim": self.ffn[0].in_features,
            "output_dim": self.ffn[-1].out_features,
            "hidden_dims": hidden_dims,
        }


class ContrastiveSupervisor(Supervisor):
    def __init__(
        self, df_x: Optional[DistanceFunction] = None, df_y: Optional[DistanceFunction] = None
    ):
        super().__init__()

        self.cont_metric = ContrastiveLoss(df_x, df_y)

    def forward(self, Z: Tensor, Y: Tensor) -> Tensor:
        mask = ~Y.isnan().any(1)

        return self.cont_metric(Z[mask], Y[mask]) if mask.any() else torch.tensor(0.0)

    def to_config(self) -> dict:
        return {"df_x": self.cont_metric.df_x, "df_y": self.cont_metric.df_y}
