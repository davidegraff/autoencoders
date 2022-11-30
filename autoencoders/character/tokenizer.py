from __future__ import annotations

from dataclasses import dataclass, asdict
import json

import re
from typing import Iterable, Iterator, Sequence

_SMILES_REGEX_PATTERN = r"(\[|\]|Br?|C[u,l]?|Zn|S[i,n]?|Li|Na?|Fe?|H|K|O|P|I|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@{1,2}|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
_SMILES_TOKENS = list("HBCNOSPFIcnosp[]()123456@-+=#/\\") + ["Cl", "Br", "@@"]


@dataclass(eq=True, frozen=True)
class SpecialTokens:
    SOS: str = "<SOS>"
    EOS: str = "<EOS>"
    PAD: str = "<PAD>"
    UNK: str = "<UNK>"

    def __iter__(self) -> Iterator[str]:
        return iter([self.SOS, self.EOS, self.PAD, self.UNK])

    def __contains__(self, t: str) -> bool:
        """is `t` a special token?"""
        return any(t == st for st in self)


class Tokenizer:
    def __init__(self, pattern: str, tokens: Iterable[str], st: SpecialTokens = SpecialTokens()):
        if any(t in st for t in tokens):
            raise ValueError("'tokens' and 'special_tokens' contain overlapping tokens!")

        tokens = sorted(tokens) + list(st)

        self.pattern = re.compile(pattern)
        self.st = st
        self.t2i = {t: i for i, t in enumerate(tokens)}
        self.i2t = {i: t for i, t in enumerate(tokens)}

    def __call__(self, s: str) -> list[str]:
        return self.encode(s)

    def __len__(self) -> int:
        """the number of the tokens in this tokenizer"""
        return len(self.t2i)

    def __contains__(self, t: str) -> bool:
        """is the token `t` in this tokenizer?"""
        return t in self.t2i

    @property
    def SOS(self) -> int:
        return self.t2i[self.st.SOS]

    @property
    def EOS(self) -> int:
        return self.t2i[self.st.EOS]

    @property
    def PAD(self) -> int:
        return self.t2i[self.st.PAD]

    @property
    def UNK(self) -> int:
        return self.t2i[self.st.UNK]

    def encode(self, word: str) -> list[int]:
        return self.tokens2ids(self.tokenize(word))

    def decode(self, ids: Sequence[int]):
        return "".join(self.ids2tokens(ids))

    def tokenize(self, word: str) -> list[str]:
        """tokenize the input word"""
        return list(self.pattern.findall(word))

    def tokens2ids(
        self, tokens: Iterable[str], add_sos: bool = True, add_eos: bool = True
    ) -> list[int]:
        ids = [self.SOS] if add_sos else []
        ids.extend([self.t2i.get(t, self.UNK) for t in tokens])
        if add_eos:
            ids.append(self.EOS)

        return ids

    def ids2tokens(
        self, ids: Sequence[int], rem_sos: bool = True, rem_eos: bool = True
    ) -> list[str]:
        if len(ids) == 0:
            return []

        if rem_sos and ids[0] == self.SOS:
            ids = ids[1:]
        if rem_eos and ids[-1] == self.EOS:
            ids = ids[:-1]

        return [self.i2t.get(i, self.st.UNK) for i in ids]

    @classmethod
    def to_json(cls, tokenzier: Tokenizer) -> str:
        return json.dumps(
            {
                "pattern": tokenzier.pattern.pattern,
                "tokens": list(t for t in tokenzier.t2i.keys() if t not in tokenzier.st),
                "st": asdict(tokenzier.st),
            }
        )

    @classmethod
    def from_json(cls, json: str):
        d = json.loads(json)
        d["st"] = SpecialTokens(**d["st"])

        return cls(**d)

    @classmethod
    def from_corpus(
        cls, pattern: str, corpus: Iterable[str], st: SpecialTokens = SpecialTokens()
    ) -> Tokenizer:
        """Build a tokenizer from the input corpus with the given tokenization scheme

        Parameters
        ----------
        pattern : str
            a regular expression defining the tokenizatio scheme
        corpus : Iterable[str]
            a set of words from which to build a vocabulary.
        st : SpecialTokens, default=SpecialTokens()
            the special tokens to use when building the vocabulary

        Returns
        -------
        Tokenizer
        """
        pattern = re.compile(pattern)
        tokens = [pattern.findall(word) for word in corpus]

        return cls(pattern.pattern, tokens, st)

    @classmethod
    def smiles_tokenizer(cls, st: SpecialTokens = SpecialTokens()) -> Tokenizer:
        """build a tokenizer for SMILES strings"""
        return cls(_SMILES_REGEX_PATTERN, _SMILES_TOKENS, st)
