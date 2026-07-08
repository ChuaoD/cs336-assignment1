import regex as re
from collections.abc import Iterable, Iterator
from .train_bpe import merge


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    
    def __init__(self, vocab, merges, special_tokens=None):
        '''
        Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.

        Args:
            vocab: dict[int, bytes]
            merges: list[tuple[bytes, bytes]]
            special_tokens: list[str] | None = None
        '''
        init_vocab_size = len(vocab)
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = {}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        if special_tokens is not None:
            # Sort the special tokens by decreasing length
            special_tokens = sorted(special_tokens, key=len, reverse=True)
            next_id = init_vocab_size
            for st in special_tokens:
                st_bytes = st.encode("utf-8")
                if st_bytes in self.inverse_vocab:
                    self.special_tokens[st] = self.inverse_vocab[st_bytes]
                else:
                    self.vocab[next_id] = st_bytes
                    self.inverse_vocab[st_bytes] = next_id
                    self.special_tokens[st] = next_id
                    next_id += 1
                    

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        '''
        Class method that constructs and returns a Tokenizer from a serialized vocabulary and list of merges and (optionally) a list of special tokens.

        Args: 
            vocab_filepath: str
            erges_filepath: str
            special_tokens: list[str] | None = None
        '''
        pass

    def _apply_merge(self, token_bytes: tuple[bytes, ...]) -> tuple[bytes, ...]:
        '''
        Apply merges to token bytes.
        '''
        while len(token_bytes) >= 2:
            pair_set = set(zip(token_bytes, token_bytes[1:]))
            # Find the first pair in merges that is in the token
            pair = next((p for p in self.merges if p in pair_set), None)
            if pair is None:
                break
            token_bytes = merge(token_bytes, pair)
        return token_bytes


    def _encode_ordinary(self, text: str) -> list[int]:
        '''
        Encode an input text without special tokens.
        '''
        indices = []
        # Pre-tokenization
        for match in re.finditer(PAT, text):
            token = match.group()
            # str -> tuple[bytes]
            token_bytes = tuple(bytes([b]) for b in token.encode("utf-8"))
            # Apply the merge
            token_bytes = self._apply_merge(token_bytes)
            # tuple[bytes] -> list[int]
            token_ids = [self.inverse_vocab[b] for b in token_bytes]
            indices.extend(token_ids)
        return indices

    def encode(self, text: str) -> list[int]:
        '''
        Encode an input text into a sequence of token IDs.
        '''
        if not self.special_tokens:
            return self._encode_ordinary(text)
        
        special_pattern = "|".join(re.escape(st) for st in self.special_tokens)
        indices = []
        pos = 0
        for match in re.finditer(special_pattern, text):
            chunk = text[pos: match.start()]
            # Encode text without special token
            indices.extend(self._encode_ordinary(chunk))
            special_token = match.group()
            # Encode the special token
            indices.append(self.special_tokens[special_token])
            pos = match.end()
        # Encode the text after the last special token
        indices.extend(self._encode_ordinary(text[pos:]))
        return indices
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        '''
        Given an iterable of strings, return a generator that lazily yields token IDs.
        '''
        for chunk in iterable:
            encoded_chunk = self.encode(chunk)
            for token_id in encoded_chunk:
                yield token_id
    
    def decode(self, ids: list[int]) -> str: 
        '''
        Decode a sequence of token IDs into text.
        '''
        text_bytes = b''.join(self.vocab[id] for id in ids)
        text = text_bytes.decode("utf-8", errors='replace')
        return text
