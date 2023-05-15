from __future__ import annotations

from abc import abstractmethod
from typing import Optional

import torch
from torch import Tensor, nn
from torch.distributions import Distribution, Normal

from ae_utils.utils import ClassRegistry, Configurable, warn_not_serializable
from ae_utils.utils.kernels import KernelFunction, KernelRegistry, InverseMultiQuadraticKernel

__all__ = ["Regularizer", "RegularizerRegistry", "DummyRegularizer", "VariationalRegularizer"]

RegularizerRegistry = ClassRegistry()


class Regularizer(nn.Module, Configurable):
    """A :class:`Regularizer` projects from the encoder output to the latent space and
    calculates the associated loss of that projection"""

    def __init__(self, d_z: int, **kwargs):
        super().__init__()

        self.d_z = d_z

    @classmethod
    @property
    @abstractmethod
    def name(self) -> str:
        """the name of the regularization loss"""

    @abstractmethod
    def setup(self, d_h: int):
        """Perform any setup necessary before using this `Regularizer`.

        NOTE: this function _must_ be called at some point in the `__init__()` function.
        """

    @abstractmethod
    def forward(self, H: Tensor) -> Tensor:
        """Project the output of the encoder into the latent space and regularize it"""

    @abstractmethod
    def train_step(self, H: Tensor) -> tuple[Tensor, Tensor]:
        """Calculate both the regularized latent representation (i.e., `forward()`) and associated
        loss of the encoder output"""

    def to_config(self) -> dict:
        return {"d_z": self.d_z}


@RegularizerRegistry.register("dummy")
class DummyRegularizer(Regularizer):
    """A :class:`DummyRegularizer` calculates no regularization loss"""

    @classmethod
    @property
    def name(self) -> str:
        return "ae"

    def setup(self, d_h: int):
        self.q_h2z_mean = nn.Linear(d_h, self.d_z)

    def forward(self, H: Tensor) -> Tensor:
        return self.q_h2z_mean(H)[None, :]

    def train_step(self, H: Tensor) -> tuple[Tensor, Tensor]:
        return self(H)[0], torch.tensor(0.0)


@RegularizerRegistry.register("vae")
class VariationalRegularizer(DummyRegularizer):
    """A :class:`VariationalRegularizer` uses the reparameterization trick to project into to the
    latent space and calculates the regularization loss as the KL divergence between the output and
    a multivariate unit normal distribution

    References
    ----------
    .. [1] Kingma, D.P.; and Welling, M.; arXiv:1312.6114v10 [stat.ML], 2014

    Notes
    -----
    This class can easily be amended to allow for arbitrary prior distributions, but (1) that
    was not necessary for the original use case and (2) makes the code scale _much_ worse in both
    `b` and `d`
    """

    def __init__(self, d_z: int):
        super().__init__(d_z)

    @classmethod
    @property
    def name(self) -> str:
        return "kl"

    def setup(self, d_h: int):
        super().setup(d_h)
        self.q_h2z_logvar = nn.Linear(d_h, self.d_z)

    def forward(self, H: Tensor) -> Tensor:
        return torch.stack((self.q_h2z_mean(H), self.q_h2z_logvar(H)))
        # Z_mean, Z_logvar = self.q_h2z_mean(H), self.q_h2z_logvar(H)

        # return self.reparameterize(Z_mean, Z_logvar)

    def train_step(self, H: Tensor) -> tuple[Tensor, Tensor]:
        Z_mean, Z_logvar = self.forward(H)
        Z = self.reparameterize(Z_mean, Z_logvar)
        # NOTE(degraff): switch following line for arbitrary priors
        l_kl = 0.5 * (Z_mean**2 + Z_logvar.exp() - 1 - Z_logvar).sum(1).mean()

        return Z, l_kl

    @staticmethod
    def reparameterize(mean: Tensor, logvar: Tensor) -> Tensor:
        sd = (logvar / 2).exp()
        eps = torch.randn_like(sd)

        return mean + eps * sd


@RegularizerRegistry.register("wae")
class WassersteinRegularizer(DummyRegularizer):
    """A :class:`WassersteinRegularizer` calculates the the regularization loss as the wasserstein
    distance between the output and a sample from an input prior distribution

    NOTE: This class does not currently support serialization with non-default values for 'kernel'
    or 'prior'

    References
    ----------
    .. [1] Tolstikhin, I.; Bousquet, O.; Gelly, S.; and Schoelkopf, B; arxiv:1711.01558 [stat.ML],
    2017
    """

    def __init__(
        self,
        d_z: int,
        kernel: KernelFunction | None = None,
        prior: Distribution | None = None,
    ):
        super().__init__(d_z)

        self.kernel = kernel or InverseMultiQuadraticKernel(2 * self.d_z)
        self.prior = prior or Normal(0, 1)

    @classmethod
    @property
    def name(self) -> str:
        return "mmd"

    def train_step(self, H: Tensor) -> tuple[Tensor, Tensor]:
        Z = self(H)[0]

        Z_prior = self.prior.sample(Z.shape).to(Z.device)
        l_mmd = self.mmd_loss(Z, Z_prior)

        return Z, l_mmd

    def MMD_loss(self, X, Y):
        K_xx = self.kernel(X, X)
        K_yy = self.kernel(Y, Y)
        K_xy = self.kernel(X, Y)

        return self.tril_mean(K_xx) + self.tril_mean(K_yy) - 2 * K_xy.mean()

    @staticmethod
    def tril_mean(A: Tensor, offset: int = -1):
        """The mean of the lower triangular (given the input offset) elements in the tensor `A`"""
        idxs = torch.tril_indices(*A.shape, offset).unbind()

        return A[idxs].mean()

    @warn_not_serializable
    def to_config(self) -> dict:
        kernel_config = {"alias": self.kernel.alias, "config": self.kernel.to_config()}

        return {"d_z": self.d_z, "kernel": kernel_config, "prior": self.prior}

    @classmethod
    def from_config(cls, config: dict) -> Configurable:
        kernel = KernelRegistry[config["kernel"]["alias"]].from_config(config["kernel"]["config"])
        config = config | {"kernel": kernel}

        return cls(**config)
