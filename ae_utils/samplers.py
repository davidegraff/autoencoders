from __future__ import annotations

from abc import abstractmethod

import torch
from torch import Tensor, nn
from torch.distributions import Distribution, Categorical
from torch.nn import functional as F

from ae_utils.utils import ClassRegistry, Configurable
from ae_utils.utils.config import warn_not_serializable

__all__ = ["Sampler", "GreedySampler", "MultinomialSampler", "NoisySampler"]

SamplerRegistry = ClassRegistry()


class Sampler(nn.Module, Configurable):
    """A `Sampler` defines the samples from a collection of unnormalized probabilities ("logits")"""

    def forward(self, logits: Tensor) -> Tensor:
        """Sample an index from last dimension of the input tensor

        Parameters
        ----------
        logits : Tensor
            a tensor of shape `... x d` containing unnormalized probabilities

        Returns
        --------
        Tensor
            a tensor of shape `...` containing the index of the selected item from the last dimension
        """
        return self.sample(logits.softmax(-1))

    @abstractmethod
    def sample(self, probs: Tensor) -> Tensor:
        """This does the same as :meth:`~.forward` but for normalized probablities.

        Depending on the subclass implementation, this may raise an error if the probabilities are not normalized
        """

    def to_config(self) -> dict:
        return {}


@SamplerRegistry.register(["greedy", "mode"])
class GreedySampler(Sampler):
    """A `GreedySampler` selects the index of the mode of the distribution"""

    def sample(self, probs: Tensor) -> Tensor:
        return probs.argmax(-1)


@SamplerRegistry.register("multinomial")
class MultinomialSampler(Sampler):
    """A `MultinomialSampler` selects an index by sampling from a multinomial ("categorical")
    distribution defined by the input probabilities"""

    def sample(self, probs: Tensor) -> Tensor:
        return Categorical(probs).sample()


@SamplerRegistry.register("topk")
class TopKSampler(MultinomialSampler):
    """A `TopKSampler` samples _only_ from the top-`k` inputs using multinomial sampling"""

    def __init__(self, k: int):
        super().__init__()

        self.k = k

    def forward(self, logits: Tensor) -> Tensor:
        logits, idxs_orig = torch.topk(logits, self.k, dim=-1)

        idxs_new = super().forward(logits)
        return idxs_orig[range(len(logits)), ..., idxs_new]


@SamplerRegistry.register("nucleus")
class NucleusSampler(MultinomialSampler):
    """A `NucleusSampler` samples from the smallest number of inputs such that the summed
    probablity is greater than or equal to an input threshold [1]_

    E.g., for a given vector of probabilities `x = [0.3, 0.2, 0.4, 0.1]` and `threshold = 0.4`, then
    nucleus sampling will sample only indices 0 and 3 (as their summed probabiliy is equal to
    `0.4`). If `threshold = 0.5`, then it will sample among indices 0, 1, and 3 (as the summed
    probability is equal to `0.6`)

    References
    ----------
    .. [1] Holtzman, A.; Buys, J., Du; L., Forbes, M.; & Choi, Y.  "The curious case of neural text degeneration." arXiv:1904.09751 [cs.CL], 2019.
    """

    def __init__(self, threshold: float):
        super().__init__()

        self.threshold = threshold

    def sample(self, probs: Tensor) -> Tensor:
        probs_sorted, idxs_orig = probs.sort(-1, descending=True)
        cdf = probs_sorted.cumsum(-1)
        mask = F.pad(cdf < self.threshold, pad=(1, 0), value=True)[:, :-1]

        idxs_new = super().sample(probs_sorted * mask)
        return idxs_orig[range(len(probs)), idxs_new]


@SamplerRegistry.register("noisy")
class NoisySampler(Sampler):
    """A `NoisySampler` adds noise sampled from the input distribution to the calculated
    probabilities before sampling based on the input :class:`~samplers.Sampler`"""

    def __init__(self, sampler: Sampler, noise: Distribution):
        self.sampler = sampler
        self.noise = noise

    def forward(self, logits: Tensor) -> Tensor:
        return self.sample(logits.softmax(-1))

    def sample(self, probs: Tensor) -> Tensor:
        return self.sampler.sample(probs + self.noise.sample(probs.shape))

    @warn_not_serializable
    def to_config(self) -> dict:
        raise {"sampler": self.sampler, "noise": self.noise}
