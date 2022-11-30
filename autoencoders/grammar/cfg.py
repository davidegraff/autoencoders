from __future__ import annotations

from collections import defaultdict
from itertools import chain
from typing import Iterator, Union

from nltk.grammar import CFG, Nonterminal
import torch
from torch import Tensor

from autoencoders.grammar.grammar import Grammar


class IndexableCFG(Grammar, CFG):
    """An `IndexableCFG` is an extension of an `nltk.grammar.CFG` that allows for index-based access
    
    NOTE: a "token" in an `nltk.grammar.CFG` is of type `Union[Nonterminal, str]`, so obtaining the index of a nonterminal token must be called like so

    >>> from nltk.grammar import Nonterminal
    >>> g: IndexableCFG
    >>> n: str # a nonterminal symbol
    >>> g[n]
    KeyError
    >>> g[Nonterminal(n)]
    [...]
    """

    def __init__(self, start, productions, calculate_leftcorners=True):
        super().__init__(start, productions, calculate_leftcorners)

        all_tokens = set()
        for p in self._productions:
            all_tokens.add(p._lhs)
            all_tokens.update(p._rhs)

        self.__i2t = dict(enumerate(all_tokens))
        self.__t2i = {beta: i for i, beta in enumerate(all_tokens)}

        self.irules = defaultdict(list)
        for p in self._productions:
            i_lhs = self.__t2i[p._lhs]
            i_rhss = [self.__t2i[t] for t in p._rhs]

            self.irules[i_lhs].append(i_rhss)
        self.irules = dict(self.irules)

    @property
    def t2i(self) -> dict[Union[Nonterminal, str], int]:
        return self.__t2i
    
    @property
    def i2t(self) -> dict[int, Union[Nonterminal, str]]:
        return self.__i2t

    @property
    def SOS(self) -> int:
        return self.__t2i[self._start]

    def PAD(self) -> int:
        return -1

    def __len__(self) -> int:
        return len(self._productions)

    def __iter__(self) -> Iterator:
        return iter(self.irules.keys())

    def __getitem__(self, idx: int) -> list[list[int]]:
        return self.irules[idx]

    def get(self, idx: int) -> list[list[int]]:
        return self.irules.get(idx, ())

    def calc_mask(self, seqs: Tensor) -> Tensor:
        mask = torch.zeros((len(seqs), len(self)), dtype=torch.bool, device=seqs.device)

        alphas = [seq[-1].item() for seq in seqs]
        for i, alpha in enumerate(alphas):
            beta_all = list(chain(*self.get(alpha)))
            mask[i][beta_all] = True

        return mask

    @classmethod
    def smiles_grammar(cls) -> IndexableCFG:
        return cls.fromstring(_SMILES_GRAMMAR)


_SMILES_GRAMMAR = """
STARTING_NON_TERMINAL -> chain
atom -> bracket_atom
atom -> aliphatic_organic
atom -> aromatic_organic
aliphatic_organic -> 'B'
aliphatic_organic -> 'C'
aliphatic_organic -> 'N'
aliphatic_organic -> 'O'
aliphatic_organic -> 'S'
aliphatic_organic -> 'P'
aliphatic_organic -> 'F'
aliphatic_organic -> 'I'
aliphatic_organic -> 'Cl'
aliphatic_organic -> 'Br'
aromatic_organic -> 'c'
aromatic_organic -> 'n'
aromatic_organic -> 'o'
aromatic_organic -> 's'
bracket_atom -> '[' BAI ']'
BAI -> isotope symbol BAC
BAI -> symbol BAC
BAI -> isotope symbol
BAI -> symbol
BAC -> chiral BAH
BAC -> BAH
BAC -> chiral
BAH -> hcount BACH
BAH -> BACH
BAH -> hcount
BACH -> charge
symbol -> aliphatic_organic
symbol -> aromatic_organic
isotope -> DIGIT
isotope -> DIGIT DIGIT
isotope -> DIGIT DIGIT DIGIT
DIGIT -> '1'
DIGIT -> '2'
DIGIT -> '3'
DIGIT -> '4'
DIGIT -> '5'
DIGIT -> '6'
DIGIT -> '7'
DIGIT -> '8'
chiral -> '@'
chiral -> '@@'
hcount -> 'H'
hcount -> 'H' DIGIT
charge -> '-'
charge -> '-' DIGIT
charge -> '-' DIGIT DIGIT
charge -> '+'
charge -> '+' DIGIT
charge -> '+' DIGIT DIGIT
bond -> '-'
bond -> '='
bond -> '#'
bond -> '/'
bond -> '\\'
ringbond -> DIGIT
ringbond -> bond DIGIT
branched_atom -> atom
branched_atom -> atom RB
branched_atom -> atom BB
branched_atom -> atom RB BB
RB -> RB ringbond
RB -> ringbond
BB -> BB branch
BB -> branch
branch -> '(' chain ')'
branch -> '(' bond chain ')'
chain -> branched_atom
chain -> chain branched_atom
chain -> chain bond branched_atom
Nothing -> None
"""
