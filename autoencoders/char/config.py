from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from torch import nn

from autoencoders.char.modules import RnnDecoder, RnnEncoder
from autoencoders.char.regularizers import Regularizer
from autoencoders.char.samplers import Sampler
from autoencoders.char.tokenizer import Tokenizer


class Config:
    pass

@dataclass
class FactoryConfig(Config):
    alias: str
    config: Config


@dataclass
class TokenizerConfig(Config):
    pattern: str
    tokens: Iterable[str]
    st: dict[str, str]


@dataclass
class EmbeddingConfig(Config):
    num_embeddings: int
    embedding_dim: int
    padding_idx: int


@dataclass
class RnnEncoderConfig(Config):
    embedding: nn.Embedding
    d_h: int
    n_layers: int
    dropout: float
    bidir: bool
    d_z: int
    regularizer: Regularizer


@dataclass
class RnnDecoderConfig(Config):
    SOS: int
    EOS: int
    embedding: nn.Embedding
    d_z: int
    d_h: int
    n_layers: int
    dropout: float
    sampler: Sampler


@dataclass
class VAEConfig(Config):
    tokenizer: Tokenizer
    encoder: RnnEncoder
    decoder: RnnDecoder
    lr: float
    v_reg: FactoryConfig
    shared_emb: bool
