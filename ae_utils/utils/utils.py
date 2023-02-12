from torch import Tensor


def safe_cross_entropy(
    logits: Tensor,
    labels: Tensor,
    ignore_index: int = -100,
    eps: float = 1e-9,
):
    probs = logits.softmax(1)
    mask = labels = ignore_index

    likelihoods = (probs[range(len(labels)), labels] + eps).log()
    likelihoods[mask] = 0

    return -likelihoods.sum()