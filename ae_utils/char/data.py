from typing import Iterable, Optional

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


class SupervisedDataset(Dataset):
    def __init__(self, dset: UnsupervisedDataset, Y: Optional[np.ndarray]):
        self.dset = dset
        if Y is not None:
            if len(Y) != len(self.dset):
                raise ValueError(
                    "args 'dset' and 'Y' must have same length! "
                    f"got: {len(dset)}, {len(Y)}, respectively."
                )
            self.Y = torch.from_numpy(Y).float()
        else:
            self.Y = torch.empty((len(dset), 1))

    def __len__(self) -> int:
        return len(self.dset)
    
    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        return self.dset[i], self.Y[i]

    @staticmethod
    def collate_fn(batch: Iterable[tuple[Tensor, Tensor]]) -> tuple[list[Tensor], Tensor]:
        idss, ys = zip(*batch)

        return idss, torch.stack(ys)
