from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn.utils import rnn

from autoencoders.samplers import ModeSampler, Sampler


class CharacterDecoder(nn.Module):
    def __init__(
        self,
        SOS: int,
        EOS: int,
        embedding: nn.Embedding,
        d_z: int = 16,
        d_h: int = 384,
        n_layers: int = 3,
        dropout: float = 0.2,
        sampler: Sampler | None = None,
    ):
        super().__init__()

        self.emb = embedding
        self.z2h = nn.Linear(d_z, d_h)
        self.rnn = nn.GRU(self.emb.embedding_dim, d_h, n_layers, batch_first=True, dropout=dropout)
        self.h2v = nn.Linear(d_h, self.emb.num_embeddings)
        self.sampler = sampler or ModeSampler()

        self.SOS = SOS
        self.EOS = EOS
        self.PAD = embedding.padding_idx
        self.d_z = d_z
        self.d_v = embedding.num_embeddings

    def forward(self, xs: Sequence[Tensor], Z: Tensor) -> Tensor:
        lengths = [len(x) for x in xs]

        X = rnn.pad_sequence(xs, batch_first=True, padding_value=self.PAD)
        X_emb = self.emb(X)
        X_packed = rnn.pack_padded_sequence(X_emb, lengths, batch_first=True, enforce_sorted=False)

        H = self.z2h(Z)
        H_0 = H.unsqueeze(0).repeat(self.rnn.num_layers, 1, 1)

        O_packed, _ = self.rnn(X_packed, H_0)
        O, _ = nn.utils.rnn.pad_packed_sequence(O_packed, True)

        return self.h2v(O)

    def generate(self, Z: Tensor, max_len: int = 80) -> list[Tensor]:
        n = len(Z)

        x_t = torch.tensor(self.SOS, device=Z.device).repeat(n)
        X_gen = torch.tensor([self.PAD], device=Z.device).repeat(n, max_len)
        X_gen[:, 0] = self.SOS

        seq_lens = torch.tensor([max_len], device=Z.device).repeat(n)
        eos_mask = torch.zeros(n, dtype=torch.bool, device=Z.device)

        H_t = self.z2h(Z).unsqueeze(0).repeat(self.rnn.num_layers, 1, 1)
        for t in range(1, max_len):
            x_emb = self.emb(x_t).unsqueeze(1)
            O, H_t = self.rnn(x_emb, H_t)
            logits = self.h2v(O.squeeze(1)).softmax(-1)

            x_t = self.sampler(logits)
            X_gen[~eos_mask, t] = x_t[~eos_mask]

            eos_mask_t = ~eos_mask & (x_t == self.EOS)
            seq_lens[eos_mask_t] = t + 1
            eos_mask = eos_mask | eos_mask_t

        return [X_gen[i, : seq_lens[i]] for i in range(len(X_gen))]
