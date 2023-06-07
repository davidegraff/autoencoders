from __future__ import annotations

from abc import abstractmethod

import torch
from torch import Tensor, nn
from torch.distributions import Distribution, Normal

from ae_utils.utils import ClassRegistry, Configurable, warn_not_serializable
from ae_utils.utils.kernels import KernelFunction, KernelRegistry, InverseMultiQuadraticKernel

__all__ = ["Regularizer", "RegularizerRegistry", "DummyRegularizer", "VariationalRegularizer"]

RegularizerRegistry = ClassRegistry()


class Regularizer(nn.Module, Configurable):
    """A :class:`Regularizer` projects from the encoder output to a distribution in the latent
    space"""

    name: str
    """the name of the regularization loss"""

    @abstractmethod
    def __len__(self) -> int:
        """the size of the latent space"""

    @abstractmethod
    def forward(self, H: Tensor) -> Tensor:
        """Project the output of the encoder into its latent distribution
        
        Parameters
        ----------
        H : Tensor
            a tensor of shape `b x d_h`, where `b` is the batch size, and `d_h` is the size encoder
            hidden representation, containing the output of the encoder

        Returns
        -------
        Z : Tensor
            a tensor of shape `* x b x d_z`, where `*` is the number of parameters in the latent
            distribution and `d_z` is the size of the latent space. `Z[0]` will always correspond
            to the mean of the output distribution
        """

    @abstractmethod
    def train_step(self, H: Tensor) -> tuple[Tensor, Tensor]:
        """Calculate both the regularized latent representation and associated regularization
        loss

        Parameters
        ----------
        H : Tensor
            a tensor of shape `b x d_h`, where `b` is the batch size, and `d_h` is the size encoder
            hidden representation, containing the output of the encoder

        Returns
        -------
        Z_reg : Tensor
            a tensor of shape `b x d_z`, where `d_z` is the size of the latent space, containing
            the regularized latent representation
        loss : Tensor
            a scalar corresponding to the regularization loss 
        """


@RegularizerRegistry.register("dummy")
class DummyRegularizer(Regularizer):
    """A :class:`DummyRegularizer` projects directly into the latent space.
     
    It places no prior on the distribution in the latent space, so it has no loss (i.e., 0). The
    output of a `DummyRegularizer` is a a tensor of shape `1 x b x d_z`
    """

    name = "ae"

    def __init__(self, d_h: int, d_z: int):
        super().__init__()

        self.q_h2z_mean = nn.Linear(d_h, d_z)

    def __len__(self) -> int:
        return self.q_h2z_mean.out_features
    
    def forward(self, H: Tensor) -> Tensor:
        return self.q_h2z_mean(H)[None, :]

    def train_step(self, H: Tensor) -> tuple[Tensor, Tensor]:
        return self(H)[0], torch.tensor(0.0)

    def to_config(self) -> dict:
        return {"d_h": self.q_h2z_mean.in_features, "d_z": len(self)}


@RegularizerRegistry.register("vae")
class VariationalRegularizer(DummyRegularizer):
    """A :class:`VariationalRegularizer` projects each point into a normal distibtribution in the latent space
    
    The regularization loss is calculated as the KL divergence between the output and a multivariate unit normal distribution via the reparameterization trick. The output of a `VariationalRegularizer` is a a tensor of shape `2 x b x d_z`, where the 0th dimension
    corresponds to the mean and variance (indices 0 and 1, respectively,) of a normal distribution
    in the latent space, `b` is the size of a batch, and `d_z` is the size of the latent dimension

    References
    ----------
    .. [1] Kingma, D.P.; and Welling, M.; arXiv:1312.6114v10 [stat.ML], 2014

    Notes
    -----
    This class can easily be amended to allow for arbitrary prior distributions, but (1) that
    was not necessary for the original use case and (2) makes the code scale _much_ worse in both
    `b` and `d`
    """

    name = "kl"

    def __init__(self, d_h: int, d_z: int):
        super().__init__(d_h, d_z)

        self.q_h2z_logvar = nn.Linear(d_h, d_z)

    def forward(self, H: Tensor) -> Tensor:
        return torch.stack((self.q_h2z_mean(H), self.q_h2z_logvar(H)))

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
    """A :class:`WassersteinRegularizer` projects directly into the latent space.
    
    It uses the input prior to calculate the the regularization loss as the wasserstein distance
    between the output and a corresponding sample from the prior. The output of a
    `WassersteinRegularizer` is a a tensor of shape `1 x b x d_z`.

    NOTE: This class does not currently support serialization with non-default values for 'kernel'
    or 'prior'

    References
    ----------
    .. [1] Tolstikhin, I.; Bousquet, O.; Gelly, S.; and Schoelkopf, B; arxiv:1711.01558 [stat.ML],
    2017
    """

    name = "mmd"

    def __init__(
        self,
        d_h: int,
        d_z: int,
        kernel: KernelFunction | None = None,
        prior: Distribution | None = None,
    ):
        super().__init__(d_h, d_z)

        self.kernel = kernel or InverseMultiQuadraticKernel(2 * len(self))
        self.prior = prior or Normal(0, 1)

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

        return super().to_config() | {"kernel": kernel_config, "prior": self.prior}

    @classmethod
    def from_config(cls, config: dict) -> Configurable:
        kernel = KernelRegistry[config["kernel"]["alias"]].from_config(config["kernel"]["config"])
        config = config | {"kernel": kernel}

        return cls(**config)
