from abc import abstractmethod
from typing import Iterable, Optional

import torch
from torch import Tensor, nn

from ae_utils.utils import Configurable, DistanceFunction, ContrastiveLoss, build_ffn
from ae_utils.utils.distances import CosineDistance, DistanceFunctionRegistry, PNormDistance
from ae_utils.utils.registry import ClassRegistry

SupervisorRegistry = ClassRegistry()


class Supervisor(nn.Module, Configurable):
    @abstractmethod
    def forward(self, Z: Tensor, Y: Tensor) -> Tensor:
        """Calculate the supervision loss term.

        NOTE: this function _internally_ handles semisupervision. I.e., the targets `Y` should
        contain *both* labeled *and* unlabeled inputs
        
        Parameters
        ----------
        Z: Tensor
            a tensor of shape `b x d`, where `b` is the batch size and `d` is the size of the
            latent space, containing latent representations
        Y: Tensor
            a tensor of shape `b x t`, where `b` is the batch size and `t` is the number of
            supervision targets, containing the target values

        Returns
        -------
        loss : Tensor
            a scalar containing the _fully reduced_ loss
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
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Optional[Iterable[int]] = None):
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
    """Supervise the learning using a contrastive loss.
     
    The loss is of the form: L = MSE(D_xx, D_yy), where `D_xx` and `D_yy` are `n x n` matrices
    containing the respctive input- and output-space distances between points `i` and `j`.
    
    Parameters
    ----------
    df_x : DistanceFunction | None
        the distance function to use in the input space. If `None`, use `CosineDistance`
    df_y : DistanceFunction | None
        the distance funtion to use in the output space. If `None`, use `PNormDistance(inf)`. I.e.,
        the infinity-norm.
    """
    def __init__(
        self, df_x: Optional[DistanceFunction] = None, df_y: Optional[DistanceFunction] = None
    ):
        super().__init__()

        df_x = df_x or CosineDistance()
        df_y = df_y or PNormDistance(torch.inf)
        
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
