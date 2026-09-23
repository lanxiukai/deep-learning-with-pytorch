"""Text tokenization and vocabulary primitives."""

import collections


def tokenize(lines, token='word'):
    """
    Split text lines into word or character tokens.
    
    Args:
        lines: a list of text lines
        token: the type of token (Default: 'word')
    Returns:
        A list of tokens (2D list [([token])])
        - If token is 'word', return a list of words
        - If token is 'char', return a list of characters
    """
    if token == 'word':
        return [line.split() for line in lines]     # 2D list [([word])]
    elif token == 'char':
        return [list(line) for line in lines]  # 2D list [([char])]
    else:
        print('Error: unknown token type: ' + token)


class Vocab:
    """
    Text vocabulary.
    
    Args:
        tokens: a list of tokens (1D list [token] or 2D list [[token]]) (Default: None)
        min_freq: the minimum frequency of the token (less than min_freq will be ignored) (Default: 0)
        reserved_tokens: a list of reserved tokens (Default: None)
    """
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # Sort by frequency
        counter = count_corpus(tokens)
        # Sorted by frequency in descending order, type: list of tuples [(token, frequency)]
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # The index of the unknown token is 0, and the reserved tokens are prepended
        # 1D list [token], the index is the position of the token in the list
        self.idx_to_token = ['<unk>'] + reserved_tokens
        # 2D dictionary {token: index}
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        """Return the number of tokens."""
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        """
        Get the index of the token (if it exists, otherwise return the index of the unknown token (0)).

        Args:
            tokens: a single token or a list of tokens (1D list [token])
        Returns:
            - If tokens is a single token, return the index of the token
            - If tokens is a list of tokens, return a list of indices of the tokens
        """
        if not isinstance(tokens, (list, tuple)):  # if tokens is a single token
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        """
        Convert indices to tokens.
        
        Args:
            indices: a single index or a list of indices (1D list [index])
        Returns:
            - If indices is a single index, return the token
            - If indices is a list of indices, return a list of tokens
        """
        if not isinstance(indices, (list, tuple)):  # if indices is a single index
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):
        """(@property) The index of the unknown token is 0."""
        return 0

    @property
    def token_freqs(self):
        """(@property) Return a list of tuples sorted by token frequencies: [(token, frequency)]."""
        return self._token_freqs


def count_corpus(tokens):
    """
    Count token frequencies.
    
    Args:
        tokens: a list of tokens (1D list [token] or 2D list [[token]])
    Returns:
        A counter of tokens (collections.Counter)
    """
    # Here tokens is a 1D list or a 2D list
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Flatten a 2D list of tokens into a single list
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)
