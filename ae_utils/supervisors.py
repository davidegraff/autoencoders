from abc import abstractmethod
from typing import Optional

import torch
from torch import Tensor, nn

from ae_utils.utils import Configurable, DistanceFunction, ContrastiveLoss, build_ffn
from ae_utils.utils.distances import DistanceFunctionRegistry
from ae_utils.utils.registry import ClassRegistry

SupervisorRegistry = ClassRegistry()


class Supervisor(nn.Module, Configurable):
    @abstractmethod
    def forward(self, Z: Tensor, Y: Tensor) -> Tensor:
        """Calculate the supervision loss term.

        NOTE: this function _internally_ handles semisupervision. I.e., the targets `Y` should
        contain *both* labeled *and* unlabeled inputs
        """

    @abstractmethod
    def check_input_dim(self, input_dim: int):
        """Check that the intended input dimension is valid for this supervisor.

        Raises
        ------
        ValueError
            if the input dimension is not valid
        """


@SupervisorRegistry.register("dummy")
class DummySupervisor(Supervisor):
    def forward(self, Z: Tensor, Y: Tensor) -> Tensor:
        return torch.tensor(0.0)

    def check_input_dim(self, d_z: int):
        return

    def to_config(self) -> dict:
        return {}


@SupervisorRegistry.register("regression")
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

    def check_input_dim(self, d_z: int):
        input_dim = self.ffn[0].in_features
        if input_dim != d_z:
            raise ValueError(f"Invalid input dimensionality! got: {d_z}. expected: {input_dim}")

    def to_config(self) -> dict:
        hidden_dims = [layer.out_features for layer in self.ffn[-1]] if len(self.ffn) > 1 else None

        return {
            "input_dim": self.ffn[0].in_features,
            "output_dim": self.ffn[-1].out_features,
            "hidden_dims": hidden_dims,
        }


@SupervisorRegistry.register("cont")
class ContrastiveSupervisor(Supervisor):
    def __init__(
        self, df_x: Optional[DistanceFunction] = None, df_y: Optional[DistanceFunction] = None
    ):
        super().__init__()

        self.cont_metric = ContrastiveLoss(df_x, df_y)

    def forward(self, Z: Tensor, Y: Tensor) -> Tensor:
        mask = Y.isfinite().any(1)

        return self.cont_metric(Z[mask], Y[mask]) if mask.any() else torch.tensor(0.0)

    def check_input_dim(self, d_z: int):
        return

    def to_config(self) -> dict:
        return {
            "df_x": {
                "alias": self.cont_metric.df_x.alias,
                "config": self.cont_metric.df_x.to_config(),
            },
            "df_y": {
                "alias": self.cont_metric.df_y.alias,
                "config": self.cont_metric.df_y.to_config(),
            },
        }

    @classmethod
    def from_config(cls, config: dict) -> Configurable:
        df_x_alias = config["df_x"]["alias"]
        df_x_config = config["df_x"]["config"]
        df_x = DistanceFunctionRegistry[df_x_alias].from_config(df_x_config)

        df_y_alias = config["df_y"]["alias"]
        df_y_config = config["df_y"]["config"]
        df_y = DistanceFunctionRegistry[df_y_alias].from_config(df_y_config)

        return cls(df_x, df_y)
