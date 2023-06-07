from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import Tensor, nn
from torch.nn.utils import rnn

from ae_utils.utils import Configurable
from ae_utils.samplers import Sampler, SamplerRegistry, GreedySampler
from ae_utils.regularizers import Regularizer, VariationalRegularizer, RegularizerRegistry

__all__ = ["RnnEncoder", "RnnDecoder"]


class RnnEncoder(nn.Module, Configurable):
    def __init__(
        self,
        embedding: nn.Embedding,
        d_h: int = 256,
        n_layers: int = 1,
        dropout: float = 0.0,
        bidir: bool = True,
        d_z: int = 128,
        regularizer: Regularizer | None = None,
    ):
        super().__init__()

        self.emb = embedding
        self.d_h = d_h
        self.rnn = nn.GRU(
            self.emb.embedding_dim,
            d_h,
            n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidir,
        )
        d_h_rnn = 2 * d_h if bidir else d_h

        self.reg = regularizer or VariationalRegularizer(d_h_rnn, d_z)

    @property
    def d_v(self) -> int:
        """The size of the input vocabulary"""
        return self.emb.num_embeddings

    @property
    def d_z(self) -> int:
        return len(self.reg)

    @property
    def PAD(self) -> int:
        return self.emb.padding_idx

    def encode(self, xs: Iterable[Tensor]) -> Tensor:
        xs_emb = [self.emb(x) for x in xs]
        X = rnn.pack_sequence(xs_emb, enforce_sorted=False)

        _, H = self.rnn(X)
        H = H[-(1 + int(self.rnn.bidirectional)) :]

        return torch.cat(H.split(1), -1).squeeze(0)

    def forward(self, xs: Sequence[Tensor]) -> Tensor:
        return self.reg(self.encode(xs))

    def train_step(self, xs: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
        return self.reg.train_step(self.encode(xs))

    def to_config(self) -> dict:
        config = {
            "embedding": {
                "num_embeddings": self.emb.num_embeddings,
                "embedding_dim": self.emb.embedding_dim,
                "padding_idx": self.emb.padding_idx,
            },
            "d_h": self.rnn.hidden_size,
            "n_layers": self.rnn.num_layers,
            "dropout": self.rnn.dropout,
            "bidir": self.rnn.bidirectional,
            "d_z": self.reg.d_z,
            "regularizer": {"alias": self.reg.alias, "config": self.reg.to_config()},
        }

        return config

    @classmethod
    def from_config(cls, config: dict) -> RnnEncoder:
        emb = nn.Embedding(**config["embedding"])
        reg_alias = config["regularizer"]["alias"]
        reg_config = config["regularizer"]["config"]
        reg = RegularizerRegistry[reg_alias].from_config(reg_config)

        config = config | dict(embedding=emb, regularizer=reg)
        return cls(**config)


class RnnDecoder(nn.Module, Configurable):
    def __init__(
        self,
        SOS: int,
        EOS: int,
        embedding: nn.Embedding,
        d_z: int = 128,
        d_h: int = 512,
        n_layers: int = 3,
        dropout: float = 0.2,
        sampler: Sampler | None = None,
    ):
        """an `RnnDecoder` autoregressively generates sequences starting from an input vector

        Parameters
        ----------
        SOS : int
            the start-of-sequence token
        EOS : int
            the end-of-sequence token
        embedding : nn.Embedding
            the embedding layer used to embed sequences of inputs into the embedding space
        d_z : int, default=128
            the dimension of the input vector
        d_h : int, default=512
            the hidden dimension of the GRU
        n_layers : int, default=3
            the number of layers in the GRU
        dropout : float, default=0.2
            the dropout rate to use during training
        sampler : Sampler | None, default=None
            the sampler to use. If None, defaults to using a `GreedySampler`
        """
        super().__init__()

        self.SOS = SOS
        self.EOS = EOS
        self.emb = embedding
        self.d_z = d_z

        self.z2h = nn.Linear(self.d_z, d_h)
        self.rnn = nn.GRU(self.emb.embedding_dim, d_h, n_layers, batch_first=True, dropout=dropout)
        self.h2v = nn.Linear(d_h, self.d_v)
        self.sampler = sampler or GreedySampler()

    @property
    def d_v(self) -> int:
        return self.emb.num_embeddings

    @property
    def PAD(self) -> int:
        return self.emb.padding_idx

    def train_step(self, xs: Sequence[Tensor], Z: Tensor) -> Tensor:
        lengths = [len(x) for x in xs]
        X = rnn.pad_sequence(xs, batch_first=True, padding_value=self.PAD)

        X_emb = self.emb(X)
        X_packed = rnn.pack_padded_sequence(X_emb, lengths, batch_first=True, enforce_sorted=False)
        H = self.z2h(Z)
        H_0 = H.unsqueeze(0).repeat(self.rnn.num_layers, 1, 1)
        O_packed, _ = self.rnn(X_packed, H_0)
        O, _ = nn.utils.rnn.pad_packed_sequence(O_packed, True)

        return self.h2v(O)

    def forward(self, Z: Tensor, max_len: int = 80) -> list[Tensor]:
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

    def to_config(self) -> dict:
        config = {
            "SOS": self.SOS,
            "EOS": self.EOS,
            "embedding": {
                "num_embeddings": self.emb.num_embeddings,
                "embedding_dim": self.emb.embedding_dim,
                "padding_idx": self.emb.padding_idx,
            },
            "d_z": self.d_z,
            "d_h": self.rnn.hidden_size,
            "n_layers": self.rnn.num_layers,
            "dropout": self.rnn.dropout,
            "sampler": self.sampler.alias,
        }

        return config

    @classmethod
    def from_config(cls, config: dict) -> RnnDecoder:
        emb = nn.Embedding(**config["embedding"])
        sampler = SamplerRegistry[config["sampler"]]()

        config = config | dict(embedding=emb, sampler=sampler)
        return cls(**config)
