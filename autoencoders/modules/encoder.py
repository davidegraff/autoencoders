from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn.utils import rnn


class Encoder(nn.Module):
    def __init__(
        self,
        embedding: nn.Embedding,
        d_emb: int = 64,
        d_h: int = 384,
        n_layers: int = 1,
        dropout: float = 0.0,
        bidir: bool = True,
        d_z: int = 16,
    ):
        super().__init__()

        self.d_z = d_z

        self.emb = embedding
        self.rnn = nn.GRU(
            d_emb, d_h, n_layers, batch_first=True, dropout=dropout, bidirectional=bidir
        )
        self.d_h_rnn = 2 * d_h if bidir else d_h
        self.q_h2z_1 = nn.Linear(self.d_h_rnn, d_z)

    def forward(self, xs: Sequence[Tensor]) -> Tensor:
        xs_emb = [self.emb(x) for x in xs]
        X = rnn.pack_sequence(xs_emb, enforce_sorted=False)

        _, H = self.rnn(X)
        H = H[-(1 + int(self.rnn.bidirectional)) :]

        return torch.cat(H.split(1), -1).squeeze(0)