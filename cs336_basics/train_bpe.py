from __future__ import annotations

import os
import regex as re
from collections import Counter
from multiprocessing import Pool
from functools import partial

from .pretokenization_example import find_chunk_boundaries

SPLIT_SPECIAL_TOKEN = b"<|endoftext|>"
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def count_adjacent_pairs(
    pre_tokens: dict[tuple[bytes, ...], int]
) -> tuple[
    dict[tuple[bytes, bytes], tuple[int, set[tuple[bytes, ...]]]],
    dict[tuple[bytes, ...], list[tuple[bytes, bytes]]]
]:
    '''
    Given pre-tokenized text,
    return a mapping from pairs to count and tokens containing the pair,
    and a mapping from tokens to contained pairs. 
    '''
    pair_info = {}
    token_pairs = {}
    for token, freq in pre_tokens.items():
        token_pairs[token] = []
        # Count the number of occurrences of each pair of tokens
        for pair in zip(token, token[1:]):
            if pair not in pair_info:
                pair_info[pair] = (0, set())
            pair_count, token_set = pair_info[pair]
            token_set.add(token)
            pair_info[pair] = (pair_count + freq, token_set)
            token_pairs[token].append(pair)
    return pair_info, token_pairs

def pre_tokenize_chunk(start_end: tuple[int, int], input_path, special_tokens):
    '''
    Pre-tokenize a chunk.
    '''
    pre_token_counts = Counter()
    start, end = start_end
    with open(input_path, "rb") as f:
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

    return pre_token_counts

def merge(
    byte_seq: tuple[bytes, ...], 
    pair: tuple[bytes, bytes]
) -> tuple[bytes, ...]:
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

    # Parallelize pre-tokenization
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes,        
                                           SPLIT_SPECIAL_TOKEN)
    chunk_ranges = list(zip(boundaries[:-1], boundaries[1:]))
    worker = partial(pre_tokenize_chunk, 
                     input_path=input_path, special_tokens=special_tokens)
    with Pool(num_processes) as pool:
        pre_token_subcounts = pool.map(worker, chunk_ranges)

    pre_token_counts = Counter()
    for subcounts in pre_token_subcounts:
        pre_token_counts.update(subcounts)

    pre_tokens: dict[tuple[bytes, ...], int] = {
        tuple(bytes([b]) for b in k.encode("utf-8")): v 
        for k, v in pre_token_counts.items()
    }
    
    # Compute BPE merges
    num_merges = vocab_size - init_vocab_size
    merges: list[tuple[bytes, bytes]] = []
    pair_info, token_pairs = count_adjacent_pairs(pre_tokens)

    for i in range(num_merges):
        # Find the pair with the highest count
        # Break ties by preferring the lexicographically greater pair
        pair = max(pair_info, key=lambda k: (pair_info[k][0], k))
        # Find the tokens containig the pair to merge
        affected_tokens = pair_info[pair][1].copy()
        for old_token in affected_tokens:
            token_freq = pre_tokens[old_token]
            # Remove the contribution of the affected token to all its pairs
            for p in token_pairs[old_token]:
                pair_count, token_set = pair_info[p]
                token_set.discard(old_token)
                pair_info[p] = (pair_count - token_freq, token_set)
            # Merge the pair in the affected token
            merged_token = merge(old_token, pair)
            pre_tokens[merged_token] = pre_tokens.pop(old_token)
            token_pairs.pop(old_token)
            token_pairs[merged_token] = []
            # Recompute pair counts for the affected token
            for p in zip(merged_token, merged_token[1:]):
                if p not in pair_info:
                    pair_info[p] = (0, set())
                pair_count, token_set = pair_info[p]
                token_set.add(merged_token)
                pair_info[p] = (pair_count + token_freq, token_set)
                token_pairs[merged_token].append(p)
        pair_info.pop(pair)  # no zero-count pairs
        merges.append(pair)
        idx = init_vocab_size + i 
        vocab[idx] = pair[0] + pair[1]

    return vocab, merges