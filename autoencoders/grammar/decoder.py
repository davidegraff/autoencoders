from collections import deque
from typing import Sequence

import torch
from torch import Tensor, nn

from autoencoders.modules import RnnDecoder
from autoencoders.grammar.grammar import Grammar
from autoencoders.samplers import Sampler


class GrammarDecoder(RnnDecoder):
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
        sampler: Sampler | None = None,
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
