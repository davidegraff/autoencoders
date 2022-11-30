from abc import abstractmethod
from collections.abc import Mapping
from typing import Any, Iterator

from torch import Tensor


class Grammar(Mapping):
    """A `Grammar` is the abstraction of a formal grammar

    A grammar G is formally a 4-tuple: G = (V, T, P, S), where:
    - V is the set of non-terminal symbols
    - T (typically :math:`\Sigma`) is the set of terminal symbols, disjoint from V
    - P is the set of production rules of the form :math:`P : T -> (T \cup V)*`, where `*` is the
    Kleene star operation [1]_. Note that in the context of rule application, the following notation is used: :math:`alpha \in P` and :math:`beta \in (T \cup N)`. I.e., :math:`alpha` and
    :math:`beta` are concrete realizations of the left- and right-hand sides of a rule.
    - S is the unique start symbol (a member of V)
    
    A `Grammar` is indexable by the _index_ of the token on the left-hand side of a rule to yield a 
    list of _all_ possible transitions on the right-hand side of the rule.
    
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Kleene_star
    .. [2] https://en.wikipedia.org/wiki/Formal_grammar
    """

    @property
    @abstractmethod
    def t2i(self) -> dict[Any, int]:
        """a mapping from a token to its index"""

    @property
    @abstractmethod
    def i2t(self) -> dict[int, Any]:
        """a mapping from an index to its token"""

    @property
    @abstractmethod
    def SOS(self) -> int:
        """the index of the start of sequence ("SOS") token"""

    @property
    @abstractmethod
    def PAD(self) -> int:
        """the padding index"""

    @abstractmethod
    def __len__(self) -> int:
        """the number of rules ("productions") in this grammar"""

    @abstractmethod
    def __iter__(self) -> Iterator:
        """Iterate over the all left-hand side token indexes. I.e., all valid inputs to
        `__getitem__`"""

    @abstractmethod
    def __getitem__(self, idx: int) -> list[list[int]]:
        """get the products of token `idx`

        Parameters
        ----------
        idx : int
            the index of the nonterminal token on the left-hand side

        Returns
        -------
        list[list[int, ...]]
            a list of lists, where the outer list is the number of rules associated with the given 
            token index, and the inner list containins the indices of the tokens produced by the 
            rule
        
        Raises
        ------
        KeyError
            if there is no rule with the given token as the left-hand side
        """

    @abstractmethod
    def get(self, idx: int) -> list[list[int]]:
        """like `__getitem__` but returns an empty tuple if the given token isn't associated with
        any rules"""


    @abstractmethod
    def calc_mask(self, seqs: Tensor) -> Tensor:
        """Calculate the invalid rule mask based on the corresponding parse trees
        
        Parameters
        ----------
        seqs : Tensor
            a tensor of shape `b x l` containing `b` sequences of length `l`

        Returns
        -------
        Tensor
            a boolean tensor of shape `b x r`, where `r` is the number of rules in this grammar
        """
