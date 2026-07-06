from __future__ import annotations

import os
import regex as re
from collections import Counter
from multiprocessing import Process

from .pretokenization_example import find_chunk_boundaries

SPLIT_SPECIAL_TOKEN = b"<|endoftext|>"
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def count_adjacent_pairs(
        byte_seq: tuple[bytes, ...],
        freq: int,
        counts: dict[tuple[bytes, ...], int] | None = None):
    '''
    Given a tuple of bytes, and its frequency in the text,
    return a dictionary of counts of consecutive pairs
    Example:
    (b'b', b'a', b'n', b'a', b'n', b'a'), 2 
    -> {(b'b', b'a'): 2, (b'a', b'n'): 4, (b'n', b'a'): 4}
    '''
    counts = {} if counts is None else counts
    for pair in zip(byte_seq, byte_seq[1:]):
        counts[pair] = counts.get(pair, 0) + freq
    return counts

def merge(byte_seq: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    '''
    Given a tuple of bytes, replace all consecutive occurrences of pair of bytes object with one bytes object.
    Example:
    (b'b', b'a', b'n', b'a', b'n', b'a'), (b'n', b'a') 
    -> (b'b', b'a', b'na', b'na')
    '''
    new_bytes = []
    i = 0
    while i < len(byte_seq):
        if (
            i < len(byte_seq) - 1
            and byte_seq[i] == pair[0]
            and byte_seq[i+1] == pair[1]
            ):
            new_bytes.append(pair[0] + pair[1])
            i += 2
        else:
            new_bytes.append(byte_seq[i])
            i += 1
    return tuple(new_bytes)

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    # Initialize vocabulary
    assert vocab_size >= 256
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}
    for i, st in enumerate(special_tokens):
        vocab[256 + i] = st.encode("utf-8")
    init_vocab_size = len(vocab)

    # Pre-tokenization
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes,        
                                           SPLIT_SPECIAL_TOKEN)
        # Parallelize pre-tokenization
        pre_token_counts = Counter()
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Remove special tokens before pre-tokenization
            subchunks = re.split("|".join(re.escape(st) for st in special_tokens), chunk)
            for subchunk in subchunks:
                # Run pre-tokenization on subchunk and store the counts for each pre-token
                pre_token_subcounts = Counter(
                    pretoken.group() for pretoken in re.finditer(PAT, subchunk)
                    )
                pre_token_counts.update(pre_token_subcounts)
    pre_tokens: dict[tuple[bytes, ...], int] = {
        tuple(bytes([b]) for b in k.encode("utf-8")): v 
        for k, v in pre_token_counts.items()
        }
    
    # Compute BPE merges
    num_merges = vocab_size - init_vocab_size
    merges: list[tuple[bytes, bytes]] = []
    
    for i in range(num_merges):
        counts = {}
        for byte_seq, freq in pre_tokens.items():
            # Count the number of occurrences of each pair of tokens
            count_adjacent_pairs(byte_seq, freq, counts)
        # Find the pair with the highest count
        # Break ties by preferring the lexicographically greater pair
        pair = max(counts.items(), key=lambda x: (x[1], x[0]))[0]
        # Merge the pair
        pre_tokens = {merge(k, pair): v for k, v in pre_tokens.items()}
        merges.append(pair)
        idx = init_vocab_size + i 
        vocab[idx] = pair[0] + pair[1]

    return vocab, merges


'''
find the most frequent pair
jump to only the words containing that pair
update only those words
fix only the pair counts affected by those changed words


pair_counts = global popularity of each pair
pair_to_words = where each pair appears
word_to_pairs = what pairs each word currently has'''