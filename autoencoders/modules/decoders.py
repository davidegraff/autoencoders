from collections import deque
from typing import Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn.utils import rnn

from autoencoders.grammars import Grammar
from autoencoders.modules.samplers import ModeSampler, Sampler


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
        sampler: Optional[Sampler] = None,
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


class GrammarDecoder(CharacterDecoder):
    def __init__(
        self,
        SOS: int,
        EOS: int,
        embedding: nn.Embedding,
        grammar: Grammar,
        d_emb: int = 64,
        d_z: int = 16,
        d_h: int = 384,
        n_layers: int = 3,
        dropout: float = 0.2,
        sampler: Optional[Sampler] = None,
    ):
        super().__init__(SOS, EOS, embedding, d_emb, d_z, d_h, n_layers, dropout, sampler)
        self.G = grammar

    def forward(self, xs: Sequence[Tensor], Z: Tensor) -> Tensor:
        logits = super().forward(xs, Z)

        masks = [
            self.G.calc_mask([x[:t + 1] for x in xs], self.d_v)
            for t in range(logits.shape[1])
        ]
        mask = torch.stack(masks, dim=1)
        logits[~mask] = -torch.inf

        return logits

    def generate(self, Z: Tensor, max_len: int = 80) -> list[Tensor]:
        n = len(Z)
        stacks = [deque([self.SOS]) for _ in range(n)]

        eos_mask = torch.zeros(n, device=Z.device, dtype=torch.bool)
        seq_lens = torch.zeros(n, device=Z.device)
        alpha = torch.empty(n, device=Z.device)
        trees = torch.tensor([self.PAD], device=Z.device).repeat(n, max_len)

        H_t = self.z2h(Z).unsqueeze(0).repeat(self.rnn.num_layers, 1, 1)
        for t in range(max_len):
            eos_mask = torch.tensor([len(s) > 0 for s in stacks])
            seq_lens[eos_mask] = t + 1
            alpha[:] = torch.tensor([s.pop() if len(s) > 0 else self.PAD for s in stacks])
            trees[:, t] = alpha

            x_emb = self.emb(alpha).unsqueeze(1)
            O, H_t = self.rnn(x_emb, H_t)
            logits = self.h2v(O.squeeze(1))

            mask = self.G.calc_mask(trees[:, : t + 1])
            logits[~mask] = -torch.inf
            rules = self.sampler(logits)

            betas = [self.G.get(rule.item()) for rule in rules]
            for stack, beta in zip(stacks, betas):
                stack.extend(beta)

        return [t[:n] for t, n in zip(trees, seq_lens)]
