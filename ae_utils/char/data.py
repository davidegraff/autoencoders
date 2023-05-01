from typing import Iterable

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

from ae_utils.char.tokenizer import Tokenizer


class UnsupervisedDataset(Dataset):
    def __init__(self, words: Iterable[str], tokenizer: Tokenizer):
        self.data = list(words)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i: int) -> Tensor:
        xs = self.tokenizer(self.data[i])

        return torch.tensor(xs, dtype=torch.long)

    @staticmethod
    def collate_fn(idss) -> list[Tensor]:
        return idss


class CachedUnsupervisedDataset(UnsupervisedDataset):
    def __init__(self, words: Iterable[str], tokenizer: Tokenizer, quiet: bool = False):
        self.data = [
            torch.tensor(tokenizer(w), dtype=torch.long)
            for w in tqdm(words, "Caching", unit="word", disable=quiet, leave=False)
        ]

    def __getitem__(self, i: int) -> Tensor:
        return self.data[i]


class SemisupervisedDataset(Dataset):
    def __init__(self, dset: UnsupervisedDataset, Y: np.ndarray, normalize: bool = True):
        if len(Y) != len(dset):
            raise ValueError(
                "args 'dset' and 'Y' must have same length! "
                f"got: {len(dset)}, {len(Y)}, respectively."
            )

        Y: Tensor = torch.from_numpy(Y).float()
        mask = torch.isfinite(Y)

        self.dset = dset
        self.Y = (Y - Y[mask].mean(0)) / Y[mask].std(0) if normalize else Y

    def __len__(self) -> int:
        return len(self.dset)

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        return self.dset[i], self.Y[i]

    @staticmethod
    def collate_fn(batch: Iterable[tuple[Tensor, Tensor]]) -> tuple[list[Tensor], Tensor]:
        idss, ys = zip(*batch)

        return idss, torch.stack(ys)
