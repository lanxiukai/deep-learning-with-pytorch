"""Time Machine dataset and sequence batching primitives."""

import random
import re

import torch

from dl_utils.data.downloads import download
from .vocabulary import Vocab, tokenize


def read_time_machine(*, data_root=None):
    """
    Load the Time Machine dataset into a list of text lines.

    Args:
        data_root: the shared dataset root passed to the D2L downloader

    Returns:
        A list of text lines (1D list [text line])
    """
    with open(download('time_machine', data_root=data_root), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


def load_corpus_time_machine(max_tokens=-1):
    """
    Load the Time Machine dataset into a list of corpus indices and a vocabulary object.
    
    Args:
        max_tokens: the maximum number of tokens to load (Default: -1, unlimited)
    Returns:
        A tuple of corpus indices and vocabulary: (corpus, vocab)
        - corpus: a list of corpus indices corresponding to the tokens (1D list [index])
        - vocab: a vocabulary object (Vocab)
    """
    lines = read_time_machine()
    tokens = tokenize(lines, 'char') or []
    vocab = Vocab(tokens)
    # Because each line in the Time Machine dataset is not necessarily a sentence or a paragraph,
    # we flatten all text lines into a single list
    corpus = [vocab[token] for line in tokens for token in line]  # hidden flatten: 1D list [index]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


def seq_data_iter_random(corpus, batch_size, num_steps):
    """
    Generate a batch of subsequences using random sampling.
    
    Args:
        corpus: a list of corpus indices corresponding to the tokens (1D list [index])
        batch_size: the batch size
        num_steps: the number of steps
    Returns:
        A batch of subsequences as tensors: (X, Y)
        - X: a tensor of shape (batch_size, num_steps)
        - Y: a tensor of shape (batch_size, num_steps)
    """
    # Partition the sequence starting from a random offset, whose range includes num_steps - 1
    corpus = corpus[random.randint(0, num_steps - 1):]
    # Subtract 1 because we need to account for the labels
    num_subseqs = (len(corpus) - 1) // num_steps
    # Initial indices of subsequences with length num_steps
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # During iteration with random sampling, subsequences from two adjacent random batches 
    # are not necessarily adjacent in the original sequence
    random.shuffle(initial_indices)

    def data(pos):
        # Return the subsequence of length num_steps starting from index pos
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # Here, initial_indices contains the random initial indices of subsequences
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        # Yield the subsequences as tensors: (batch_size, num_steps)
        yield torch.tensor(X), torch.tensor(Y)


def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """
    Generate a batch of subsequences using sequential partitioning.
    
    Args:
        corpus: a list of corpus indices corresponding to the tokens (1D list [index])
        batch_size: the batch size
        num_steps: the number of steps
    Returns:
        A batch of subsequences as tensors: (X, Y)
        - X: a tensor of shape (batch_size, num_steps)
        - Y: a tensor of shape (batch_size, num_steps)
    """
    # Split the sequence starting from a random offset
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        # Yield the subsequences as tensors: (batch_size, num_steps)
        yield X, Y


class SeqDataLoader:
    """
    Iterator for loading sequence data.
    
    Args:
        batch_size: the batch size
        num_steps: the number of steps
        use_random_iter: whether to use random sampling
        max_tokens: the maximum number of tokens to load
    """
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        """Return the iterator of the data."""
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    """
    Load the Time Machine dataset into an iterator and a vocabulary object.
    
    Args:
        batch_size: the batch size
        num_steps: the number of steps
        use_random_iter: whether to use random sampling (Default: False)
        max_tokens: the maximum number of tokens to load (Default: 10000)
    Returns:
        A tuple of an iterator and a vocabulary object: (data_iter, data_iter.vocab)
        - data_iter: an iterator for loading sequence data (SeqDataLoader)
        - data_iter.vocab: a vocabulary object (Vocab)
    """
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab
