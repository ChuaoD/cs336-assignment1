from __future__ import annotations
import argparse
import regex as re
from collections.abc import Iterator
from pathlib import Path
import numpy as np
import time

from cs336_basics.tokenizer import Tokenizer


DEFAULT_SPECIAL_TOKENS = ["<|endoftext|>"]
DEFAULT_READ_SIZE = 1024 * 1024
DEFAULT_PROGRESS_INTERVAL = 1000

def iter_chunks(
    input_path: Path,
    special_tokens: list[str] = DEFAULT_SPECIAL_TOKENS,
    read_size: int = DEFAULT_READ_SIZE,
) -> Iterator[str]:
    """Yield text chunks ending at recognized special-token boundaries.

    The input is read incrementally. Each yielded chunk includes the special
    token at its end, so concatenating the encoded chunks preserves all tokens.

    A possible match near the end of a read block is deferred until the next
    block. This prevents a shorter overlapping special token from being emitted
    before enough text is available to detect a longer special token.
    """
    special_tokens = sorted(
        set(special_tokens),
        key=len,
        reverse=True
    )
    special_pattern = re.compile("|".join(re.escape(token) for token in special_tokens))
    max_special_token_length = max(map(len, special_tokens))

    buffer = ""

    with input_path.open("r", encoding="utf-8", errors="strict") as input_file:
        while True:
            block = input_file.read(read_size)
            reached_eof = block == ""

            if reached_eof:
                break

            buffer += block

            # Split at special tokens inside the buffer
            while buffer:
                match = special_pattern.search(buffer)
                if match is None:
                    break
                if match.start() > len(buffer) - max_special_token_length:
                    break
                yield buffer[:match.end()]
                buffer = buffer[match.end():]
        if buffer:
            yield buffer

def encode_data(
    input_path: Path,
    output_path: Path,
    tokenizer: Tokenizer,
    special_tokens: list[str],
    dtype: np.dtype,
    read_size: int = DEFAULT_READ_SIZE
) -> int:
    """Encode a corpus into a Numpy array of token ids.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    token_count = 0
    chunk_number = 0

    print(f"Encoding {input_path}")

    with output_path.open("wb") as output_file:
        for chunk_number, chunk in enumerate(
            iter_chunks(input_path, special_tokens, read_size),
            start=1
            ):
            token_ids = tokenizer.encode(chunk)
            token_array = np.asarray(token_ids, dtype=dtype)
            token_array.tofile(output_file)
            token_count += int(token_array.size)

            if chunk_number % DEFAULT_PROGRESS_INTERVAL == 0:
                print(
                    f"  Encoded {chunk_number:,} chunks; "
                    f"{token_count:,} tokens"
                )
    if token_count == 0:
        raise ValueError(f"Tokenization produced no tokens for {input_path}.")

    print(
        f"Finished {output_path}: {token_count:,} tokens across "
        f"{chunk_number:,} chunks, dtype={dtype.name}"
    )
    return token_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Encode training and validation text corpora into memory-mapped "
            ".bin token-ID arrays."
        )
    )
    
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--vocab_path", type=Path, required=True)
    parser.add_argument("--merges_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument(
        "--special_tokens",
        type=str,
        nargs="+",
        default=DEFAULT_SPECIAL_TOKENS,
        help=(
            "One or more special tokens recognized by the tokenizer. "
            "Example: --special_tokens '<|endoftext|>' '<|padding|>'"
        ),
    )
    parser.add_argument(
        "--read_size",
        type=int,
        default=DEFAULT_READ_SIZE,
        help="Approximate number of text characters read at once.",
    )
    parser.add_argument(
        "--storage_dtype",
        type=str,
        default="uint16"
    )

    args = parser.parse_args()

    special_tokens = args.special_tokens

    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.vocab_path,
        merges_filepath=args.merges_path,
        special_tokens=special_tokens,
    )

    storage_dtype = np.dtype(args.storage_dtype)

    print("Tokenizer loaded")
    print(f"Vocabulary size:  {len(tokenizer.vocab):,}")
    print(f"Maximum token ID: {max(tokenizer.vocab):,}")
    print(f"Storage dtype:    {storage_dtype.name}")
    print(f"Special tokens:   {special_tokens}")

    start_time = time.perf_counter()
    token_count = encode_data(
        args.input_path,
        args.output_path,
        tokenizer,
        special_tokens,
        dtype=storage_dtype,
        read_size=args.read_size
    )
    elapsed_time = time.perf_counter() - start_time
    tokens_per_second = token_count / elapsed_time
    print(
        f"\nEncoding time: {elapsed_time:.2f} seconds\n"
        f"Encoding speed: {tokens_per_second:,.0f} tokens/second"
    )

    # Validate encoded file
    token_ids = np.memmap(
        args.output_path,
        dtype=storage_dtype,
        mode="r",
    )

    assert len(token_ids) == token_count

    minimum_id = int(token_ids.min())
    maximum_id = int(token_ids.max())
    assert minimum_id >= min(tokenizer.vocab)
    assert maximum_id <= max(tokenizer.vocab)
    del token_ids
    print("\nEncoding completed")


if __name__ == "__main__":
    main()