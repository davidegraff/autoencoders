from itertools import chain
from typing import Iterable, Optional

from torch import Tensor, nn


def safe_cross_entropy(logits: Tensor, labels: Tensor, ignore_index: int = -100, eps: float = 1e-9):
    probs = logits.softmax(1)
    mask = labels = ignore_index

    likelihoods = (probs[range(len(labels)), labels] + eps).log()
    likelihoods[mask] = 0

    return -likelihoods.sum()


def build_ffn(
    input_dim: int,
    output_dim: int,
    hidden_dims: Optional[Iterable[int]] = None,
    bias: bool = False,
    dropout: float = 0.0,
) -> nn.Sequential:
    hidden_dims = list(hidden_dims or [])
    sizes = [input_dim, *hidden_dims, output_dim]

    layers = [
        (nn.Linear(d1, d2, bias), nn.Dropout(dropout), nn.ReLU())
        for d1, d2 in zip(sizes[:-1], sizes[1:])
    ]
    layers = list(chain(*layers))

    return nn.Sequential(*layers[:-2])
